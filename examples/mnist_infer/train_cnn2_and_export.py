#!/usr/bin/env python3
"""
Train a 2-conv CNN on MNIST, save state_dict for aricode-pack.

Architecture (chosen so all spatial sizes stay at 28×28 across both
conv layers, which is what arr_f32_conv2d_3x3_p1 currently supports):

    Conv2d(1, 8,  kernel=3, padding=1)    conv1: (8, 1, 3, 3)
    ReLU
    Conv2d(8, 16, kernel=3, padding=1)    conv2: (16, 8, 3, 3)  ← multi-channel
    ReLU
    MaxPool2d(2)                          → (16, 14, 14)
    Flatten                               → 3136
    Linear(3136, 64)                      fc1: (64, 3136)
    ReLU
    Linear(64, 10)                        fc2: (10, 64)

This stresses the pack tool's multi-channel conv path (the second
conv has C_in = 8) and emits a deeper feature stack than the v0.4
demo did.
"""

from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


HERE       = Path(__file__).resolve().parent
DATA_DIR   = HERE / "mnist_data"
WEIGHTS_PT = HERE / "cnn2_mnist.pt"
BATCH      = 128
EPOCHS     = 8
LR         = 1e-3


class CNN2(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1,  8, 3, padding=1)
        self.conv2 = nn.Conv2d(8, 16, 3, padding=1)
        self.fc1   = nn.Linear(16 * 14 * 14, 64)
        self.fc2   = nn.Linear(64, 10)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        x = x.flatten(1)
        x = F.relu(self.fc1(x))
        return self.fc2(x)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    tfm = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,)),
    ])
    train_ds = datasets.MNIST(DATA_DIR, train=True,  download=True, transform=tfm)
    test_ds  = datasets.MNIST(DATA_DIR, train=False, download=True, transform=tfm)
    train_dl = DataLoader(train_ds, batch_size=BATCH, shuffle=True,  num_workers=2)
    test_dl  = DataLoader(test_ds,  batch_size=512, shuffle=False, num_workers=2)

    model = CNN2().to(device)
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
