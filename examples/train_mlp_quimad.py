"""
QUIMAD vs Adam — Multimodal Regression Demo

Trains the same MLP with Adam and QIMADTorch on a synthetic 2D regression
task whose loss landscape has many local minima (Rastrigin-shaped targets).
Shows convergence curves and final MSE for each optimizer.

Run from the project root:
    python examples/train_mlp_quimad.py
"""

import math
import sys
from pathlib import Path

import matplotlib.pyplot as plt
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from quimad_torch import QIMADTorch


# -- Data: multimodal target with many local minima --------------------------─

torch.manual_seed(42)
N = 400
X = torch.rand(N, 2) * 4 - 2  # [-2, 2]²
y = (
    X[:, 0] ** 2 + X[:, 1] ** 2
    - 1.5 * torch.cos(2 * math.pi * X[:, 0])
    - 1.5 * torch.cos(2 * math.pi * X[:, 1])
).unsqueeze(1)
y = y + 0.05 * torch.randn_like(y)


# -- Model factory ------------------------------------------------------------─

def make_model():
    torch.manual_seed(0)
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )


# -- Generic training loop ----------------------------------------------------─

def train(model, opt, epochs=300):
    criterion = nn.MSELoss()
    losses = []
    is_quimad = isinstance(opt, QIMADTorch)

    for epoch in range(epochs):
        if is_quimad:
            def closure():
                opt.zero_grad()
                loss = criterion(model(X), y)
                loss.backward()
                return loss
            loss_val = opt.step(closure)
        else:
            opt.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            opt.step()
            loss_val = loss.item()

        losses.append(loss_val)
        if (epoch + 1) % 50 == 0:
            print(f"  epoch {epoch+1:3d}: loss = {loss_val:.5f}")

    return losses


# -- Run comparison ------------------------------------------------------------

EPOCHS = 300

print("-- Adam ------------------------------------------------------------------")
adam_model = make_model()
adam_opt = torch.optim.Adam(adam_model.parameters(), lr=0.01)
adam_losses = train(adam_model, adam_opt, EPOCHS)

print("\n-- QIMADTorch (8 agents) -------------------------------------------------")
quimad_model = make_model()
quimad_opt = QIMADTorch(quimad_model.parameters(), num_agents=8, eta=0.01, seed=42)
quimad_losses = train(quimad_model, quimad_opt, EPOCHS)

print(f"\nFinal MSE  Adam: {adam_losses[-1]:.5f} | QIMADTorch: {quimad_losses[-1]:.5f}")

# -- Plot ----------------------------------------------------------------------

fig, ax = plt.subplots(figsize=(9, 4))
ax.semilogy(adam_losses, label="Adam (lr=0.01)", color="royalblue", linewidth=1.5)
ax.semilogy(quimad_losses, label="QIMADTorch (8 agents, η=0.01)", color="darkorange", linewidth=1.5)
ax.set_xlabel("Epoch")
ax.set_ylabel("MSE Loss (log scale)")
ax.set_title("QUIMAD vs Adam — Multimodal Regression")
ax.legend()
ax.grid(True, alpha=0.3)
plt.tight_layout()

out_path = Path(__file__).parent.parent / "results" / "quimad_torch_convergence.png"
out_path.parent.mkdir(exist_ok=True)
plt.savefig(out_path, dpi=150)
print(f"Plot saved: {out_path}")
plt.show()
