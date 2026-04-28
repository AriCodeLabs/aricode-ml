#!/usr/bin/env python3
"""
Train a small CNN on MNIST in PyTorch (GPU when available), then save
the state_dict for aricode-pack to bundle into a static binary.

Architecture (chosen to match aricode's existing AVX2 builtins
exactly — no transpose, no shape-shuffle at pack time):

    Conv2d(1, 8, kernel=3, padding=1)   conv weight (8, 1, 3, 3)
    ReLU
    MaxPool2d(2)                         output (8, 14, 14)
    Flatten                              output 1568
    Linear(1568, 64)                     fc1 weight (64, 1568)
    ReLU
    Linear(64, 10)                       fc2 weight (10, 64)

  ~101 K parameters, ~405 KB of f32 weights.

This is the f32 mirror of aricode-ml/examples/mnist/mnist_cnn_f32.ari
that already trains end-to-end inside aricode itself; here we just
let PyTorch + the GPU do the work, then ship the result for inference.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


HERE      = Path(__file__).resolve().parent
DATA_DIR  = HERE / "mnist_data"
WEIGHTS_PT = HERE / "cnn_mnist.pt"
BATCH     = 128
EPOCHS    = 10
LR        = 1e-3


class CNN(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=3, padding=1, bias=True)
        self.fc1  = nn.Linear(8 * 14 * 14, 64)
        self.fc2  = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),  # match aricode demos
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

    model = CNN().to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-3)

    t0 = time.time()
    for epoch in range(EPOCHS):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = F.cross_entropy(model(xb), yb)
            loss.backward()
            opt.step()
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
        acc = correct / len(test_ds)
        print(f"epoch {epoch+1:2d}/{EPOCHS}  acc={acc:.4f}  ({time.time()-t0:.1f}s)")

    torch.save(model.state_dict(), WEIGHTS_PT)
    print(f"wrote {WEIGHTS_PT}")
    print("\nstate_dict shapes:")
    for k, v in model.state_dict().items():
        print(f"  {k}: {tuple(v.shape)}")


if __name__ == "__main__":
    main()
