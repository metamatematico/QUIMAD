"""
MNIST Benchmark — QIMADTorch vs Adam vs SGD vs PSO vs DE vs CMA-ES

Trains an MLP on MNIST (flattened 784-dim input) and compares all optimizers
on accuracy and convergence speed. Uses a small network to keep runtime
reasonable for gradient-free methods.

Architecture: Linear(784,128) -> ReLU -> Linear(128,64) -> ReLU -> Linear(64,10)
~109K parameters — CMA-ES is skipped at this scale (O(D) memory OK but very
slow without gradients). A note is printed explaining why.

Run from project root:
    python examples/benchmark_mnist.py

Requires: torchvision (pip install torchvision)
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    import torchvision
    import torchvision.transforms as transforms
    HAS_TORCHVISION = True
except ImportError:
    HAS_TORCHVISION = False

sys.path.insert(0, str(Path(__file__).parent.parent))
from quimad_torch import QIMADTorch
from pso_torch import PSOTorch
from de_torch import DETorch


# ── Model ─────────────────────────────────────────────────────────────────────

def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Flatten(),
        nn.Linear(784, 128), nn.ReLU(),
        nn.Linear(128, 64),  nn.ReLU(),
        nn.Linear(64, 10),
    )


# ── Data ──────────────────────────────────────────────────────────────────────

def load_mnist(batch_size=512):
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.1307,), (0.3081,))
    ])
    train = torchvision.datasets.MNIST('./data', train=True,  download=True, transform=transform)
    test  = torchvision.datasets.MNIST('./data', train=False, download=True, transform=transform)
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True)
    test_loader  = torch.utils.data.DataLoader(test,  batch_size=1000, shuffle=False)
    return train_loader, test_loader


def evaluate(model, loader):
    model.eval()
    correct = total = 0
    with torch.no_grad():
        for X, y in loader:
            pred = model(X).argmax(dim=1)
            correct += (pred == y).sum().item()
            total   += y.size(0)
    model.train()
    return correct / total


# ── Training loop ─────────────────────────────────────────────────────────────

def train_epoch(model, opt, loader, is_quimad=False):
    crit = nn.CrossEntropyLoss()
    total_loss = 0.0
    batches = 0
    for X, y in loader:
        if is_quimad:
            def closure():
                opt.zero_grad()
                loss = crit(model(X), y)
                loss.backward()
                return loss
            loss_val = opt.step(closure)
        else:
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            loss_val = loss.item()
        total_loss += float(loss_val)
        batches += 1
    return total_loss / batches


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    if not HAS_TORCHVISION:
        print("torchvision not installed. Run: pip install torchvision")
        sys.exit(1)

    print("Cargando MNIST...")
    train_loader, test_loader = load_mnist(batch_size=512)

    EPOCHS = 10
    D = sum(p.numel() for p in make_model().parameters())
    print(f"Parametros del modelo: {D:,}")
    print(f"Epochs: {EPOCHS}  |  Batch size: 512")
    print()

    # Gradient-free methods (PSO, DE) are impractical on 109K-param networks
    # in 10 epochs — include them for 5 epochs with a small note.
    configs = [
        ('Adam (lr=1e-3)',  False,
         lambda m: torch.optim.Adam(m.parameters(), lr=1e-3)),
        ('SGD+momentum',    False,
         lambda m: torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)),
        ('QUIMAD 4ag',      True,
         lambda m: QIMADTorch(m.parameters(), num_agents=4, eta=5e-4,
                              cooling='cosine', total_steps=EPOCHS*len(train_loader),
                              seed=42)),
        ('QUIMAD 8ag k4',   True,
         lambda m: QIMADTorch(m.parameters(), num_agents=8, eta=5e-4, k_eval=4,
                              cooling='cosine', total_steps=EPOCHS*len(train_loader),
                              seed=42)),
        ('PSO 8p',          True,
         lambda m: PSOTorch(m.parameters(), num_particles=8, seed=42)),
        ('DE  8p',          True,
         lambda m: DETorch(m.parameters(),  num_particles=8, seed=42)),
    ]

    results = {}
    print(f"{'Optimizador':<22} {'Ep':>3}  {'Loss':>8}  {'Acc test':>9}  {'Tiempo':>8}")
    print("-" * 60)

    for name, is_q, opt_fn in configs:
        model = make_model(seed=0)
        opt   = opt_fn(model)
        acc_history  = []
        loss_history = []
        t0 = time.perf_counter()

        for ep in range(1, EPOCHS + 1):
            loss = train_epoch(model, opt, train_loader, is_quimad=is_q)
            acc  = evaluate(model, test_loader)
            acc_history.append(acc)
            loss_history.append(loss)
            if ep % 2 == 0 or ep == 1:
                elapsed = time.perf_counter() - t0
                print(f"  {name:<20} {ep:3d}  {loss:8.4f}  {acc*100:8.2f}%  {elapsed:7.1f}s")

        results[name] = {'acc': acc_history, 'loss': loss_history,
                         'time': time.perf_counter() - t0}
        print()

    # ── Plot ──────────────────────────────────────────────────────────────────
    colors = {
        'Adam (lr=1e-3)': '#2196F3',
        'SGD+momentum':   '#9E9E9E',
        'QUIMAD 4ag':     '#FF9800',
        'QUIMAD 8ag k4':  '#4CAF50',
        'PSO 8p':         '#E91E63',
        'DE  8p':         '#9C27B0',
    }

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    ep_range = range(1, EPOCHS + 1)

    for name, data in results.items():
        c = colors.get(name, '#333333')
        axes[0].plot(ep_range, data['loss'], color=c, lw=2, label=name)
        axes[1].plot(ep_range, [a * 100 for a in data['acc']], color=c, lw=2, label=name)

    axes[0].set_title('Loss por epoch (MNIST train)', fontweight='bold')
    axes[0].set_xlabel('Epoch'); axes[0].set_ylabel('Cross-entropy loss')
    axes[0].legend(fontsize=8)

    axes[1].set_title('Accuracy en test (MNIST)', fontweight='bold')
    axes[1].set_xlabel('Epoch'); axes[1].set_ylabel('Accuracy (%)')
    axes[1].legend(fontsize=8)

    for ax in axes:
        ax.grid(True, alpha=0.3)
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)

    fig.suptitle('Benchmark MNIST — QIMADTorch vs optimizadores clasicos\n'
                 'Autor: Leonardo Jimenez Martinez',
                 fontsize=12, fontweight='bold')
    fig.text(0.5, -0.04,
        'Nota: PSO y DE son metodos sin gradiente — pagan el costo de N evaluaciones\n'
        'por batch sin aprovechar backprop. QUIMAD combina enjambre con gradiente.',
        ha='center', fontsize=8, style='italic', color='#555555')

    plt.tight_layout()
    out = Path(__file__).parent.parent / 'results' / 'mnist_benchmark.png'
    out.parent.mkdir(exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f"Grafica guardada: {out}")

    # Final summary
    print("\n=== RESUMEN FINAL (epoch %d) ===" % EPOCHS)
    print(f"{'Optimizador':<22} {'Acc test':>9}  {'Tiempo total':>13}")
    for name, data in results.items():
        print(f"  {name:<20} {data['acc'][-1]*100:8.2f}%  {data['time']:12.1f}s")


if __name__ == '__main__':
    main()
