#!/usr/bin/env python3
"""
Train a 784 → 64 → 10 ReLU MLP on MNIST in PyTorch (GPU when available),
then export the weights as a raw little-endian f32 blob for aricode
to load and serve at inference time.

Blob layout (matches arr_f32_matvec's m×n row-major weight convention):
    W1 : (64, 784)   row-major  ← 50176 floats
    b1 : (64,)                  ← 64
    W2 : (10, 64)    row-major  ← 640
    b2 : (10,)                  ← 10
    -----------------------------
    Total: 50890 floats  =  203560 bytes

aricode reads the blob with three plain `arr_f32_from_file` user-fn
calls in the same order; no struct, no header, no version byte —
keep the contract minimal so the inference demo is self-explanatory.
"""

import struct
import time
from pathlib import Path

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms


HERE      = Path(__file__).resolve().parent
DATA_DIR  = HERE / "mnist_data"
WEIGHTS   = HERE / "mlp_784_64_10.f32"
N_HID     = 64
N_OUT     = 10
N_IN      = 784
BATCH     = 128
EPOCHS    = 10
LR        = 1e-3


class MLP(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(N_IN, N_HID)
        self.fc2 = nn.Linear(N_HID, N_OUT)

    def forward(self, x):
        return self.fc2(F.relu(self.fc1(x.flatten(1))))


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

    model = MLP().to(device)
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
        # eval
        model.eval()
        correct = 0
        with torch.no_grad():
            for xb, yb in test_dl:
                xb, yb = xb.to(device), yb.to(device)
                pred = model(xb).argmax(1)
                correct += (pred == yb).sum().item()
        acc = correct / len(test_ds)
        print(f"epoch {epoch+1:2d}/{EPOCHS}  acc={acc:.4f}  ({time.time()-t0:.1f}s)")

    # Export weights as raw little-endian f32 in the order aricode expects.
    # nn.Linear stores weight as (out_features, in_features) — already
    # the row-major (m, n) convention arr_f32_matvec wants.
    W1 = model.fc1.weight.detach().cpu().numpy().astype("float32")  # (64, 784)
    b1 = model.fc1.bias  .detach().cpu().numpy().astype("float32")  # (64,)
    W2 = model.fc2.weight.detach().cpu().numpy().astype("float32")  # (10, 64)
    b2 = model.fc2.bias  .detach().cpu().numpy().astype("float32")  # (10,)

    with open(WEIGHTS, "wb") as f:
        for arr in (W1, b1, W2, b2):
            arr.tofile(f)

    n_floats = W1.size + b1.size + W2.size + b2.size
    print(f"wrote {WEIGHTS}  ({n_floats} f32, {n_floats*4} bytes)")

    # Also save the state_dict so aricode-pack can pick it up.
    state_path = HERE / "mlp_784_64_10.pt"
    torch.save(model.state_dict(), state_path)
    print(f"wrote {state_path}")


if __name__ == "__main__":
    main()
