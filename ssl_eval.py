import matplotlib.pyplot as plt
import numpy as np
import torch

CIFAR10_CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
                   "dog", "frog", "horse", "ship", "truck"]
STL10_CLASSES = ["airplane", "bird", "car", "cat", "deer",
                 "dog", "horse", "monkey", "ship", "truck"]
STL10_TO_CIFAR10 = {0: 0, 1: 2, 2: 1, 3: 3, 4: 4, 5: 5, 6: 7, 8: 8, 9: 9}


def images_to_device(images, device):
    """
    Move a batch of images (tensor or tuple of views) to a device.

    Parameters
    ----------
    images : torch.Tensor or sequence of torch.Tensor
        A batch, or several augmented views of a batch (SimCLR).
    device : str or torch.device
        Target device.

    Returns
    -------
    torch.Tensor or list of torch.Tensor
        Batch on ``device``.
    """
    if isinstance(images, (list, tuple)):
        return [x.to(device, non_blocking=True) for x in images]
    return images.to(device, non_blocking=True)


@torch.no_grad()
def collect_logits(model, classifier, loader, device):
    """
    Run a frozen self-supervised model and its linear classifier on a loader.

    Works with the ``deepul_helper`` task models, whose forward pass
    returns ``(metrics, representations)``.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained self-supervised model in evaluation mode.
    classifier : torch.nn.Module
        Linear classifier trained on the frozen representations.
    loader : torch.utils.data.DataLoader
        Batches of ``(images, labels)``.
    device : str or torch.device
        Device.

    Returns
    -------
    logits : torch.Tensor
        Tensor of shape ``(n_samples, n_classes)`` on CPU.
    labels : torch.Tensor
        Ground-truth labels on CPU.
    """
    all_logits, all_labels = [], []
    for images, labels in loader:
        _, zs = model(images_to_device(images, device))
        all_logits.append(classifier(zs).float().cpu())
        all_labels.append(labels)
    return torch.cat(all_logits), torch.cat(all_labels)


def topk_accuracy(logits, labels, ks=(1, 5)):
    """
    Compute top-k accuracies in percent.

    Parameters
    ----------
    logits : torch.Tensor
        Scores of shape ``(n_samples, n_classes)``.
    labels : torch.Tensor
        Ground-truth labels.
    ks : tuple of int, default=(1, 5)
        Values of k.

    Returns
    -------
    dict
        ``{"top1": float, "top5": float, ...}``.
    """
    top = logits.topk(max(ks), dim=1).indices
    hits = top.eq(labels.view(-1, 1))
    return {f"top{k}": hits[:, :k].any(dim=1).float().mean().item() * 100 for k in ks}


def per_class_accuracy(preds, labels, n_classes):
    """
    Compute the accuracy of each class in percent.

    Parameters
    ----------
    preds : array-like of int
        Predicted labels.
    labels : array-like of int
        Ground-truth labels.
    n_classes : int
        Number of classes.

    Returns
    -------
    numpy.ndarray
        Accuracy per class; ``nan`` for classes absent from ``labels``.
    """
    preds, labels = np.asarray(preds), np.asarray(labels)
    acc = np.full(n_classes, np.nan)
    for c in range(n_classes):
        mask = labels == c
        if mask.any():
            acc[c] = (preds[mask] == c).mean() * 100
    return acc


def evaluate_frozen_model(model, classifier, loader, device, n_classes=10):
    """
    Evaluate a frozen model + linear classifier.

    Parameters
    ----------
    model : torch.nn.Module
        Pretrained self-supervised model.
    classifier : torch.nn.Module
        Linear classifier.
    loader : torch.utils.data.DataLoader
        Evaluation batches.
    device : str or torch.device
        Device.
    n_classes : int, default=10
        Number of classes.

    Returns
    -------
    dict
        ``top1``, ``top5`` and ``per_class`` (numpy array).
    """
    logits, labels = collect_logits(model, classifier, loader, device)
    metrics = topk_accuracy(logits, labels)
    metrics["per_class"] = per_class_accuracy(logits.argmax(1).numpy(), labels.numpy(), n_classes)
    return metrics


def stl10_to_cifar10_label(label):
    """
    Map an STL-10 label to the matching CIFAR-10 label.

    The two datasets share 9 classes but in a different order; STL-10's
    ``monkey`` has no CIFAR-10 equivalent (and CIFAR-10's ``frog`` none in
    STL-10).

    Parameters
    ----------
    label : int
        STL-10 label.

    Returns
    -------
    int
        CIFAR-10 label, or ``-1`` for ``monkey``.
    """
    return STL10_TO_CIFAR10.get(int(label), -1)


def stl10_transfer_loader(task, root="./data", split="test", batch_size=128, num_workers=2):
    """
    Build an STL-10 loader usable with classifiers trained on CIFAR-10.

    Images go through the CIFAR-10 *evaluation* transform of the task (no
    augmentation, same normalisation as in training), after resizing to
    the CIFAR-10 resolution when needed. Labels are remapped to CIFAR-10
    indices and the ``monkey`` class is removed.

    Parameters
    ----------
    task : {"context_encoder", "rotation", "simclr"}
        Self-supervised task, which defines the input format.
    root : str, default="./data"
        Download directory.
    split : {"train", "test"}, default="test"
        STL-10 labelled split.
    batch_size : int, default=128
        Batch size.
    num_workers : int, default=2
        Loading processes.

    Returns
    -------
    torch.utils.data.DataLoader
        Batches of ``(images, cifar10_labels)``.
    """
    from deepul_helper.data import get_transform
    from torchvision import datasets, transforms

    base = get_transform("cifar10", task, train=False)
    transform = base if task == "context_encoder" else transforms.Compose([transforms.Resize(32), base])
    dataset = datasets.STL10(root=root, split=split, transform=transform,
                             target_transform=stl10_to_cifar10_label, download=True)
    keep = np.flatnonzero(dataset.labels != STL10_CLASSES.index("monkey"))
    return torch.utils.data.DataLoader(torch.utils.data.Subset(dataset, keep),
                                       batch_size=batch_size, shuffle=False,
                                       num_workers=num_workers)


def _finish(fig, save_path, show):
    """
    Save and/or display a figure, then close it.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to finalise.
    save_path : str or None
        Output path; nothing is saved if None.
    show : bool
        Whether to call ``plt.show()``.

    Returns
    -------
    matplotlib.figure.Figure
        The same figure.
    """
    fig.tight_layout()
    if save_path:
        fig.savefig(save_path, bbox_inches="tight", dpi=150)
    if show:
        plt.show()
    else:
        plt.close(fig)
    return fig


def plot_grouped_bars(values, groups, ylabel="Accuracy (%)", title="", ylim=None,
                      save_path=None, show=True):
    """
    Draw a grouped bar chart with value labels.

    Parameters
    ----------
    values : dict
        ``series_name -> list of values`` (one value per group).
    groups : list of str
        Group labels on the x axis.
    ylabel : str, default="Accuracy (%)"
        Y-axis label.
    title : str, default=""
        Figure title.
    ylim : tuple of float, optional
        Y-axis limits.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    x = np.arange(len(groups))
    width = 0.8 / len(values)
    fig, ax = plt.subplots(figsize=(max(8, 1.1 * len(groups) * len(values)), 5))
    for i, (name, vals) in enumerate(values.items()):
        bars = ax.bar(x + (i - (len(values) - 1) / 2) * width, vals, width, label=name)
        ax.bar_label(bars, fmt="%.1f", padding=2, fontsize=8)
    ax.set_xticks(x, groups, rotation=30 if len(groups) > 5 else 0)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    if ylim:
        ax.set_ylim(*ylim)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)
    return _finish(fig, save_path, show)


def plot_history(history, title="", save_path=None, show=True):
    """
    Plot loss and accuracy curves recorded during training.

    Parameters
    ----------
    history : list of dict
        One dict per epoch with keys ``train_loss``, ``test_loss``,
        ``train_acc`` and ``test_acc``.
    title : str, default=""
        Figure title.
    save_path : str, optional
        Where to save the figure.
    show : bool, default=True
        Whether to display the figure.

    Returns
    -------
    matplotlib.figure.Figure
        The created figure.
    """
    epochs = np.arange(1, len(history) + 1)
    fig, axs = plt.subplots(1, 2, figsize=(12, 4.5))
    for ax, metric, label in zip(axs, ("loss", "acc"), ("Loss", "Accuracy (%)")):
        for split in ("train", "test"):
            ax.plot(epochs, [h[f"{split}_{metric}"] for h in history], label=split)
        ax.set_xlabel("Epoch")
        ax.set_ylabel(label)
        ax.grid(alpha=0.3)
        ax.legend()
    fig.suptitle(title)
    return _finish(fig, save_path, show)
