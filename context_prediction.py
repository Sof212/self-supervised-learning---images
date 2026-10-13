import argparse
import json
import os
import random
from functools import partial

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as TF
from torchvision import datasets

from ssl_eval import per_class_accuracy, plot_history

NEIGHBOR_OFFSETS = [(-1, -1), (-1, 0), (-1, 1),
                    (0, -1), (0, 1),
                    (1, -1), (1, 0), (1, 1)]
POSITION_NAMES = ["top-left", "top", "top-right", "left",
                  "right", "bottom-left", "bottom", "bottom-right"]
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def sample_patch_pair(image, patch_size=24, gap=8, jitter=4, rng=random):
    """
    Sample a center patch and one of its 8 neighbours (Doersch et al., 2015).

    A 3x3 grid of patches separated by ``gap`` pixels is placed inside the
    image; every patch is then shifted by up to ``jitter`` pixels. The gap
    and the jitter prevent the network from solving the task by matching
    edges or lines across patch borders.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    patch_size : int, default=24
        Side of each square patch in pixels.
    gap : int, default=8
        Space between neighbouring patches.
    jitter : int, default=4
        Maximum random shift of each patch.
    rng : random.Random, default=random
        Random generator.

    Returns
    -------
    center : PIL.Image.Image
        Center patch.
    neighbor : PIL.Image.Image
        Neighbouring patch.
    label : int
        Index in ``NEIGHBOR_OFFSETS`` of the neighbour's position.

    Raises
    ------
    ValueError
        If the grid does not fit in the image.
    """
    width, height = image.size
    extent = 3 * patch_size + 2 * gap
    if extent + 2 * jitter > min(width, height):
        raise ValueError(f"Grid of {extent}px (+{jitter}px jitter) does not fit in "
                         f"a {width}x{height} image.")
    grid_y = rng.randint(jitter, height - extent - jitter)
    grid_x = rng.randint(jitter, width - extent - jitter)
    label = rng.randrange(len(NEIGHBOR_OFFSETS))
    dy, dx = NEIGHBOR_OFFSETS[label]

    def crop(row, col):
        top = grid_y + row * (patch_size + gap) + rng.randint(-jitter, jitter)
        left = grid_x + col * (patch_size + gap) + rng.randint(-jitter, jitter)
        return image.crop((left, top, left + patch_size, top + patch_size))

    return crop(1, 1), crop(1 + dy, 1 + dx), label


def patch_to_tensor(patch, input_size, grayscale):
    """
    Resize, optionally convert to grayscale, and normalise a patch.

    Parameters
    ----------
    patch : PIL.Image.Image
        RGB patch.
    input_size : int
        Side of the network input.
    grayscale : bool
        Replace colours by luminance (repeated on 3 channels), which
        removes chromatic aberration cues.

    Returns
    -------
    torch.Tensor
        Normalised tensor of shape ``(3, input_size, input_size)``.
    """
    patch = TF.resize(patch, [input_size, input_size])
    if grayscale:
        patch = TF.to_grayscale(patch, num_output_channels=3)
    return TF.normalize(TF.to_tensor(patch), IMAGENET_MEAN, IMAGENET_STD)


def patch_pair_transform(image, patch_size=24, gap=8, jitter=4, input_size=96,
                         grayscale_prob=0.0):
    """
    Turn an image into a training example of the context prediction task.

    Unlike the original notebook, no random horizontal flip is applied to
    the patches: flipping one patch independently would make left and
    right positions ambiguous.

    Parameters
    ----------
    image : PIL.Image.Image
        Input image.
    patch_size, gap, jitter : int
        See :func:`sample_patch_pair`.
    input_size : int, default=96
        Side of the network input after resizing.
    grayscale_prob : float, default=0.0
        Probability of converting both patches to grayscale.

    Returns
    -------
    tuple
        ``(center_tensor, neighbor_tensor, label)``.
    """
    center, neighbor, label = sample_patch_pair(image.convert("RGB"), patch_size, gap, jitter)
    gray = random.random() < grayscale_prob
    return patch_to_tensor(center, input_size, gray), patch_to_tensor(neighbor, input_size, gray), label


def make_pretext_loader(root, split, batch_size=128, num_workers=4, shuffle=True, **patch_kwargs):
    """
    Build a DataLoader of patch pairs from STL-10 images.

    Parameters
    ----------
    root : str
        Download directory.
    split : str
        STL-10 split: ``"train"``, ``"test"``, ``"unlabeled"`` or
        ``"train+unlabeled"``. Labels are not used.
    batch_size : int, default=128
        Batch size.
    num_workers : int, default=4
        Loading processes.
    shuffle : bool, default=True
        Shuffle every epoch.
    **patch_kwargs
        Passed to :func:`patch_pair_transform`.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of ``((centers, neighbors, positions), _)``.
    """
    dataset = datasets.STL10(root=root, split=split, download=True,
                             transform=partial(patch_pair_transform, **patch_kwargs))
    return torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=shuffle,
                                       num_workers=num_workers, drop_last=shuffle)


def build_siamese_alexnet(n_positions=len(NEIGHBOR_OFFSETS), feature_dim=4096):
    """
    Create the siamese AlexNet-style network used for context prediction.

    Parameters
    ----------
    n_positions : int, default=8
        Number of relative positions to classify.
    feature_dim : int, default=4096
        Size of the per-patch representation (``fc6``).

    Returns
    -------
    torch.nn.ModuleDict
        ``cnn`` and ``fc6`` (shared patch encoder) and ``head`` (position
        classifier on the concatenated pair).
    """
    return nn.ModuleDict({
        "cnn": nn.Sequential(
            nn.Conv2d(3, 96, kernel_size=11, stride=4), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.LocalResponseNorm(5),
            nn.Conv2d(96, 384, kernel_size=5, stride=2, padding=2), nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2), nn.LocalResponseNorm(5),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 384, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(384),
            nn.Conv2d(384, 256, kernel_size=3, padding=1), nn.ReLU(inplace=True),
            nn.BatchNorm2d(256),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            nn.AdaptiveAvgPool2d(1), nn.Flatten(),
        ),
        "fc6": nn.Sequential(nn.Linear(256, feature_dim), nn.ReLU(inplace=True),
                             nn.BatchNorm1d(feature_dim)),
        "head": nn.Sequential(
            nn.Linear(2 * feature_dim, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, 4096), nn.ReLU(inplace=True),
            nn.Linear(4096, n_positions),
        ),
    })


def encode(model, images):
    """
    Compute the patch / image representation (``fc6`` output).

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Network from :func:`build_siamese_alexnet`.
    images : torch.Tensor
        Batch of shape ``(batch, 3, H, W)``.

    Returns
    -------
    torch.Tensor
        Representations of shape ``(batch, feature_dim)``.
    """
    return model["fc6"](model["cnn"](images))


def pretext_forward(model, centers, neighbors):
    """
    Predict the relative position of each neighbour w.r.t. its center patch.

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Network from :func:`build_siamese_alexnet`.
    centers, neighbors : torch.Tensor
        Batches of patches of shape ``(batch, 3, H, W)``.

    Returns
    -------
    torch.Tensor
        Position logits of shape ``(batch, n_positions)``.
    """
    return model["head"](torch.cat([encode(model, centers), encode(model, neighbors)], dim=1))


def run_pretext_epoch(model, loader, device, optimizer=None):
    """
    Run one epoch of the pretext task (training if an optimizer is given).

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Siamese network.
    loader : torch.utils.data.DataLoader
        Loader from :func:`make_pretext_loader`.
    device : str or torch.device
        Device.
    optimizer : torch.optim.Optimizer, optional
        If None, the epoch is run in evaluation mode without gradients.

    Returns
    -------
    loss : float
        Mean cross-entropy.
    acc : float
        Position accuracy in percent (chance level: 12.5%).
    """
    training = optimizer is not None
    model.train(training)
    total_loss, correct, total = 0.0, 0, 0
    with torch.set_grad_enabled(training):
        for (centers, neighbors, positions), _ in loader:
            centers, neighbors = centers.to(device), neighbors.to(device)
            positions = positions.to(device)
            logits = pretext_forward(model, centers, neighbors)
            loss = F.cross_entropy(logits, positions)
            if training:
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            total_loss += loss.item() * positions.size(0)
            correct += (logits.argmax(1) == positions).sum().item()
            total += positions.size(0)
    return total_loss / total, 100.0 * correct / total


def train_pretext(model, train_loader, test_loader, device, epochs=100, lr=0.006,
                  momentum=0.9, checkpoint_path=None):
    """
    Train the network on the context prediction task.

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Siamese network.
    train_loader, test_loader : torch.utils.data.DataLoader
        Pretext loaders.
    device : str or torch.device
        Device.
    epochs : int, default=100
        Number of epochs.
    lr : float, default=0.006
        SGD learning rate (cosine-annealed).
    momentum : float, default=0.9
        SGD momentum.
    checkpoint_path : str, optional
        Where to save the weights with the best test accuracy.

    Returns
    -------
    list of dict
        Per-epoch ``train_loss``, ``train_acc``, ``test_loss``,
        ``test_acc``.
    """
    optimizer = torch.optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=5e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs)
    history, best = [], -1.0
    for epoch in range(1, epochs + 1):
        train_loss, train_acc = run_pretext_epoch(model, train_loader, device, optimizer)
        test_loss, test_acc = run_pretext_epoch(model, test_loader, device)
        scheduler.step()
        history.append({"train_loss": train_loss, "train_acc": train_acc,
                        "test_loss": test_loss, "test_acc": test_acc})
        print(f"[pretext] epoch {epoch}/{epochs} | train {train_loss:.4f} / {train_acc:.2f}% "
              f"| test {test_loss:.4f} / {test_acc:.2f}%")
        if checkpoint_path and test_acc > best:
            best = test_acc
            torch.save(model.state_dict(), checkpoint_path)
    return history


def make_probe_loaders(dataset="cifar10", root="./data", input_size=96, batch_size=256,
                       num_workers=4):
    """
    Build labelled loaders for linear evaluation of the representation.

    Images are resized to ``input_size`` (no augmentation, since features
    are extracted once).

    Parameters
    ----------
    dataset : {"cifar10", "stl10"}, default="cifar10"
        Downstream dataset.
    root : str, default="./data"
        Download directory.
    input_size : int, default=96
        Network input size.
    batch_size : int, default=256
        Batch size.
    num_workers : int, default=4
        Loading processes.

    Returns
    -------
    train_loader, test_loader : torch.utils.data.DataLoader
        Labelled loaders.
    classes : list of str
        Class names.
    """
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.Resize((input_size, input_size)), transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    if dataset == "cifar10":
        train = datasets.CIFAR10(root, train=True, transform=transform, download=True)
        test = datasets.CIFAR10(root, train=False, transform=transform, download=True)
    else:
        train = datasets.STL10(root, split="train", transform=transform, download=True)
        test = datasets.STL10(root, split="test", transform=transform, download=True)
    loader = partial(torch.utils.data.DataLoader, batch_size=batch_size, shuffle=False,
                     num_workers=num_workers)
    return loader(train), loader(test), list(train.classes)


@torch.no_grad()
def extract_features(model, loader, device):
    """
    Extract frozen representations for a labelled dataset.

    Parameters
    ----------
    model : torch.nn.ModuleDict
        Trained siamese network (only the patch encoder is used).
    loader : torch.utils.data.DataLoader
        Batches of ``(images, labels)``.
    device : str or torch.device
        Device.

    Returns
    -------
    features : torch.Tensor
        Tensor of shape ``(n_samples, feature_dim)`` on CPU.
    labels : torch.Tensor
        Labels on CPU.
    """
    model.eval()
    feats, labels = [], []
    for images, y in loader:
        feats.append(encode(model, images.to(device)).cpu())
        labels.append(y)
    return torch.cat(feats), torch.cat(labels)


def train_linear_probe(train_x, train_y, test_x, test_y, n_classes, device, epochs=50,
                       lr=1e-3, batch_size=256):
    """
    Train a linear classifier on frozen, standardised features.

    Parameters
    ----------
    train_x, test_x : torch.Tensor
        Features of shape ``(n, feature_dim)``.
    train_y, test_y : torch.Tensor
        Labels.
    n_classes : int
        Number of classes.
    device : str or torch.device
        Device.
    epochs : int, default=50
        Number of epochs.
    lr : float, default=1e-3
        Adam learning rate.
    batch_size : int, default=256
        Batch size.

    Returns
    -------
    history : list of dict
        Per-epoch ``train_loss``, ``train_acc``, ``test_loss``,
        ``test_acc``.
    test_preds : numpy.ndarray
        Test predictions of the last epoch.
    """
    mean, std = train_x.mean(0), train_x.std(0) + 1e-6
    train_x, test_x = ((train_x - mean) / std).to(device), ((test_x - mean) / std).to(device)
    train_y, test_y = train_y.to(device), test_y.to(device)
    classifier = nn.Linear(train_x.shape[1], n_classes).to(device)
    optimizer = torch.optim.Adam(classifier.parameters(), lr=lr, weight_decay=1e-4)

    history = []
    for epoch in range(1, epochs + 1):
        classifier.train()
        perm = torch.randperm(len(train_x), device=device)
        for i in range(0, len(perm), batch_size):
            idx = perm[i:i + batch_size]
            loss = F.cross_entropy(classifier(train_x[idx]), train_y[idx])
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        classifier.eval()
        with torch.no_grad():
            tr_logits, te_logits = classifier(train_x), classifier(test_x)
            history.append({
                "train_loss": F.cross_entropy(tr_logits, train_y).item(),
                "train_acc": (tr_logits.argmax(1) == train_y).float().mean().item() * 100,
                "test_loss": F.cross_entropy(te_logits, test_y).item(),
                "test_acc": (te_logits.argmax(1) == test_y).float().mean().item() * 100,
            })
    print(f"[probe] final test accuracy: {history[-1]['test_acc']:.2f}%")
    return history, te_logits.argmax(1).cpu().numpy()


def parse_args():
    """
    Parse command-line arguments.

    Returns
    -------
    argparse.Namespace
        Parsed arguments.
    """
    parser = argparse.ArgumentParser(description="Context prediction (Doersch et al., 2015) on STL-10.")
    parser.add_argument("--data-root", default="./data")
    parser.add_argument("--pretext-split", default="train",
                        help="STL-10 split for the pretext task, e.g. train or train+unlabeled.")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--lr", type=float, default=0.006)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--patch-size", type=int, default=24)
    parser.add_argument("--gap", type=int, default=8)
    parser.add_argument("--jitter", type=int, default=4)
    parser.add_argument("--grayscale-prob", type=float, default=0.0)
    parser.add_argument("--probe-dataset", choices=["cifar10", "stl10"], default="cifar10")
    parser.add_argument("--probe-epochs", type=int, default=50)
    parser.add_argument("--num-workers", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-dir", default="outputs/context_prediction")
    return parser.parse_args()


def main():
    """
    Train the pretext task, then evaluate the representation with a linear probe.

    Returns
    -------
    None
    """
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    os.makedirs(args.output_dir, exist_ok=True)

    patch_kwargs = dict(patch_size=args.patch_size, gap=args.gap, jitter=args.jitter,
                        grayscale_prob=args.grayscale_prob)
    train_loader = make_pretext_loader(args.data_root, args.pretext_split, args.batch_size,
                                       args.num_workers, shuffle=True, **patch_kwargs)
    test_loader = make_pretext_loader(args.data_root, "test", args.batch_size,
                                      args.num_workers, shuffle=False, **patch_kwargs)

    model = build_siamese_alexnet().to(device)
    ckpt = os.path.join(args.output_dir, "siamese_alexnet.pth")
    pretext_history = train_pretext(model, train_loader, test_loader, device, args.epochs,
                                    args.lr, checkpoint_path=ckpt)
    plot_history(pretext_history, "Pretext task: relative patch position",
                 os.path.join(args.output_dir, "pretext_history.png"), show=False)

    model.load_state_dict(torch.load(ckpt, map_location=device))
    probe_train, probe_test, classes = make_probe_loaders(args.probe_dataset, args.data_root,
                                                          num_workers=args.num_workers)
    train_x, train_y = extract_features(model, probe_train, device)
    test_x, test_y = extract_features(model, probe_test, device)
    probe_history, preds = train_linear_probe(train_x, train_y, test_x, test_y, len(classes),
                                              device, args.probe_epochs)
    plot_history(probe_history, f"Linear probe on {args.probe_dataset}",
                 os.path.join(args.output_dir, "probe_history.png"), show=False)

    results = {
        "pretext_best_test_acc": max(h["test_acc"] for h in pretext_history),
        "probe_dataset": args.probe_dataset,
        "probe_test_acc": probe_history[-1]["test_acc"],
        "probe_per_class_acc": dict(zip(classes, per_class_accuracy(preds, test_y.numpy(),
                                                                    len(classes)).tolist())),
        "args": vars(args),
    }
    with open(os.path.join(args.output_dir, "results.json"), "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)


if __name__ == "__main__":
    main()
