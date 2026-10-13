import random

import numpy as np
import torch
from PIL import Image

from context_prediction import (NEIGHBOR_OFFSETS, build_siamese_alexnet, encode,
                                patch_pair_transform, pretext_forward, sample_patch_pair)
from ssl_eval import (CIFAR10_CLASSES, STL10_CLASSES, per_class_accuracy,
                      stl10_to_cifar10_label, topk_accuracy)


def coordinate_image(size=96):
    """
    Build an image whose red/green channels encode the pixel row/column.

    Parameters
    ----------
    size : int, default=96
        Image side.

    Returns
    -------
    PIL.Image.Image
        RGB image.
    """
    rows, cols = np.meshgrid(np.arange(size), np.arange(size), indexing="ij")
    arr = np.stack([rows, cols, np.zeros_like(rows)], axis=-1).astype(np.uint8)
    return Image.fromarray(arr)


def test_patch_pair_matches_label():
    """
    The neighbour is exactly one grid step away in the labelled direction.
    """
    image, rng = coordinate_image(), random.Random(0)
    for _ in range(50):
        center, neighbor, label = sample_patch_pair(image, patch_size=24, gap=8, jitter=0, rng=rng)
        c, n = np.asarray(center), np.asarray(neighbor)
        dy, dx = NEIGHBOR_OFFSETS[label]
        assert int(n[0, 0, 0]) - int(c[0, 0, 0]) == dy * 32
        assert int(n[0, 0, 1]) - int(c[0, 0, 1]) == dx * 32


def test_patch_pair_direction_with_jitter():
    """
    With jitter, the relative direction is still unambiguous.
    """
    image, rng = coordinate_image(), random.Random(1)
    for _ in range(50):
        center, neighbor, label = sample_patch_pair(image, jitter=4, rng=rng)
        c, n = np.asarray(center).astype(int), np.asarray(neighbor).astype(int)
        for axis, offset in enumerate(NEIGHBOR_OFFSETS[label]):
            diff = n[0, 0, axis] - c[0, 0, axis]
            if offset == 0:
                assert abs(diff) <= 8
            else:
                assert np.sign(diff) == offset and abs(diff) >= 32 - 8


def test_model_shapes():
    """
    The siamese network returns 8 position logits and fc6 features.
    """
    model = build_siamese_alexnet().eval()
    c, n, _ = patch_pair_transform(coordinate_image())
    with torch.no_grad():
        assert pretext_forward(model, c[None], n[None]).shape == (1, 8)
        assert encode(model, c[None]).shape == (1, 4096)


def test_stl10_to_cifar10_mapping():
    """
    Shared classes map to the same name; monkey is dropped.
    """
    for stl_idx, name in enumerate(STL10_CLASSES):
        cifar_idx = stl10_to_cifar10_label(stl_idx)
        if name == "monkey":
            assert cifar_idx == -1
        else:
            expected = "automobile" if name == "car" else name
            assert CIFAR10_CLASSES[cifar_idx] == expected


def test_accuracy_helpers():
    """
    Top-k and per-class accuracies on a toy example.
    """
    logits = torch.tensor([[0.9, 0.1, 0.0], [0.2, 0.3, 0.5], [0.1, 0.8, 0.1]])
    labels = torch.tensor([0, 1, 1])
    acc = topk_accuracy(logits, labels, ks=(1, 2))
    assert round(acc["top1"], 2) == 66.67 and acc["top2"] == 100.0
    per_class = per_class_accuracy([0, 2, 1], [0, 1, 1], 3)
    assert per_class[0] == 100.0 and per_class[1] == 50.0 and np.isnan(per_class[2])
