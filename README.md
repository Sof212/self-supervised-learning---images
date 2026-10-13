# Self-Supervised Visual Representation Learning

Exploring how far networks can learn useful image features **without labels**:

1. **Comparison of three pretext tasks**: Context Encoder, Rotation prediction and SimCLR,
   evaluated with linear classifiers and nearest neighbours on CIFAR-10, then under a domain shift
   to STL-10.
2. **Shortcuts**: low-level cues (chromatic aberration, colour histograms) that let a network
   "cheat" on its pretext task, and the augmentations that prevent it.
3. **Context prediction implemented from scratch** (Doersch et al., 2015): a siamese AlexNet
   trained to predict the relative position of two patches on STL-10, evaluated by linear probing.

## Results

**Linear evaluation of frozen representations on CIFAR-10 (test set)**

| Method | Top-1 | Top-5 |
|--------|------:|------:|
| Context Encoder | 45.8 % | 90.3 % |
| Rotation prediction | 79.9 % | 99.1 % |
| **SimCLR** | **92.8 %** | **99.9 %** |

**Transfer of the CIFAR-10 classifiers to STL-10**: *to be filled after running notebook 01.*

**Context prediction (own implementation)**: *to be updated after re-running notebook 03.* The
first version reached 32.7 % on the 8-way pretext task (chance: 12.5 %) and 36.9 % linear-probe
accuracy on CIFAR-10, before the fixes listed below.

<!-- Add figures from outputs/ to docs/ and display them here, e.g. ![](docs/cifar10_per_class.png) -->

## Repository

```
├── notebooks/
│   ├── 01_ssl_methods_comparison.ipynb   # pretrained Context Encoder / Rotation / SimCLR
│   ├── 02_trivial_shortcuts.ipynb        # chromatic aberration, colour histograms
│   └── 03_context_prediction.ipynb       # my implementation of context prediction
├── context_prediction.py   # patch sampling, siamese AlexNet, pretext training, linear probe (+ CLI)
├── ssl_eval.py             # evaluation helpers: top-k, per-class accuracy, STL-10 transfer, plots
├── tests/test_ssl.py
└── requirements.txt
```

## Running

The notebooks are designed for **Google Colab with a GPU**: each one starts with a setup cell
that clones what it needs. Notebook 01 downloads ~6.5 GB of pretrained weights and data provided
with the course.

Locally:

```bash
pip install -r requirements.txt
pytest
python context_prediction.py --epochs 100 --probe-dataset cifar10
python context_prediction.py --pretext-split train+unlabeled --grayscale-prob 0.3
```

## Implementation notes

**Context prediction.** A 3×3 grid of 24-px patches separated by 8-px gaps is placed in each
96×96 STL-10 image, and every patch is jittered by up to 4 px. The center patch and one random
neighbour are upsampled to 96×96 and fed to a shared AlexNet-style encoder; the concatenated
`fc6` features are classified into the 8 possible positions. The linear probe is trained on
standardised `fc6` features extracted once from the frozen encoder.

**Fixes compared with the first notebook version**

- *Patch geometry*: the 3×3 grid (3 × 40 px + 2 × 10 px = 140 px) did not fit in a 96-px image, so
  most neighbours were clipped to the border and **no longer matched their position label**. The
  grid now always fits (unit-tested).
- *Augmentation*: random horizontal flips were applied to each patch independently, which made
  left / right positions ambiguous. They were removed.
- *STL-10 transfer*: STL-10 labels were used directly with CIFAR-10 classifiers although the two
  datasets order their classes differently (bird and car swapped, horse shifted, monkey absent
  from CIFAR-10), which mechanically capped accuracy. Labels are now remapped and monkey images
  dropped; the evaluation transform (normalisation, no augmentation) now matches training.
- *Shortcut demos*: the grayscale image was computed from the colour-dropped image, color
  dropping kept the blue channel, and the "after jitter" histograms were computed on the original
  image. All three are fixed and vectorised.

## Credits

Pretrained models, the `deepul_helper` package and the original demo notebook come from the
[Berkeley CS294-158 Deep Unsupervised Learning](https://sites.google.com/view/berkeley-cs294-158-sp20/home)
course ([wilson1yan/cs294-158-ssl](https://github.com/wilson1yan/cs294-158-ssl)), adapted by
Camille Kurtz for the *Master IA – Vision et Machines Intelligentes* (Université Paris Cité).
The model comparison, STL-10 transfer study and the context prediction implementation are my work.

- Pathak et al., *Context Encoders: Feature Learning by Inpainting*, CVPR 2016
- Gidaris et al., *Unsupervised Representation Learning by Predicting Image Rotations*, ICLR 2018
- Chen et al., *A Simple Framework for Contrastive Learning of Visual Representations*, ICML 2020
- Doersch et al., *Unsupervised Visual Representation Learning by Context Prediction*, ICCV 2015
