# AlexNet Implementation (PyTorch)

A clean and beginner-friendly PyTorch implementation of an **AlexNet-style CNN** trained on **CIFAR-10**.

This repository includes:
- A model definition (`models/alexnet.py`)
- A training script (`train.py`)
- An evaluation script (`test.py`)
- Shared training/evaluation utilities (`utils/train_utils.py`)

## Project Structure

```text
.
├── models/
│   └── alexnet.py          # AlexNet-style architecture
├── utils/
│   └── train_utils.py      # train() and evaluate() helper functions
├── train.py                # Trains on CIFAR-10 and saves weights
├── test.py                 # Loads saved weights and reports test accuracy
├── alexnet_paper.pdf       # Reference paper document
├── LICENSE
└── README.md
```

## Model Overview

The model in `models/alexnet.py` uses:
- 5 convolutional layers with ReLU activations
- Max pooling after selected conv blocks
- A 3-layer fully connected classifier with dropout
- `num_classes=10` by default (for CIFAR-10)

Input images are expected in **CIFAR-10 format** (`3 x 32 x 32`).

## Requirements

- Python 3.8+
- PyTorch
- Torchvision
- tqdm

Install dependencies:

```bash
pip install torch torchvision tqdm
```

> If you want CUDA acceleration, install a CUDA-enabled PyTorch build from the official PyTorch site.

## How to Train

Run:

```bash
python train.py
```

What `train.py` does:
1. Downloads CIFAR-10 into `./data` (if not already present)
2. Creates training and test data loaders
3. Trains `AlexNet` for 10 epochs using Adam (`lr=0.001`)
4. Prints epoch loss and test accuracy
5. Saves model weights to:

```text
alexnet_cifar10.pth
```

## How to Test

After training, evaluate the saved model with:

```bash
python test.py
```

This script loads `alexnet_cifar10.pth` and prints final test accuracy.

## Notes

- Device selection is automatic (`cuda` if available, else `cpu`).
- If `alexnet_cifar10.pth` is missing, run training first.
- You can change the number of classes by editing `AlexNet(num_classes=...)`.

## Possible Improvements

- Add command-line arguments (epochs, batch size, learning rate, save path)
- Add checkpointing and resume support
- Add training curves (loss/accuracy plots)
- Add data augmentation for better generalization
- Add support for other datasets


