"""
QUIMAD Interactive Space — Hugging Face Gradio App

Autor: Leonardo Jiménez Martínez — Centro de Biomatemáticas BIOMAT

Permite comparar QIMADTorch contra Adam, SGD, PSO, DE y CMA-ES de forma
interactiva: el usuario elige tarea, optimizadores, hiperparámetros y epochs,
y obtiene curvas de convergencia en tiempo real.
"""

import math
import time

import gradio as gr
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from quimad_torch import QIMADTorch
from pso_torch    import PSOTorch
from de_torch     import DETorch
from cmaes_torch  import CMAESTorch


# ── Paleta ────────────────────────────────────────────────────────────────────

COLORS = {
    'Adam':         '#2196F3',
    'SGD':          '#9E9E9E',
    'PSO':          '#E91E63',
    'DE':           '#9C27B0',
    'CMA-ES':       '#FF5722',
    'QUIMAD 1ag':   '#FF9800',
    'QUIMAD 4ag':   '#8BC34A',
    'QUIMAD 8ag':   '#4CAF50',
}

# ── Datos ─────────────────────────────────────────────────────────────────────

def make_data(task: str, seed: int = 42):
    torch.manual_seed(seed)
    if task == 'Convexa (regresion lineal)':
        X = torch.randn(200, 4)
        y = (X @ torch.tensor([1., -1., 0.5, -0.5])).unsqueeze(1)
        return X, y, 'mse'
    elif task == 'Multimodal (paisaje con valles)':
        X = torch.rand(200, 2) * 4 - 2
        y = (X[:,0]**2 + X[:,1]**2
             - 1.5*torch.cos(2*math.pi*X[:,0])
             - 1.5*torch.cos(2*math.pi*X[:,1])).unsqueeze(1)
        return X, y, 'mse'
    else:  # Clasificacion binaria
        X = torch.randn(300, 4)
        y = ((X[:,0] + X[:,1]**2 - X[:,2] * X[:,3]) > 0).float().unsqueeze(1)
        return X, y, 'bce'


def make_model(task: str, seed: int = 0):
    torch.manual_seed(seed)
    if 'Convexa' in task:
        return nn.Linear(4, 1)
    elif 'Multimodal' in task:
        return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))
    else:
        return nn.Sequential(nn.Linear(4, 16), nn.ReLU(), nn.Linear(16, 1), nn.Sigmoid())


def make_optimizer(name: str, model, num_agents: int, eta: float, seed: int, epochs: int):
    p = model.parameters()
    if name == 'Adam':
        return torch.optim.Adam(p, lr=eta), False
    if name == 'SGD':
        return torch.optim.SGD(p, lr=eta, momentum=0.9), False
    if name == 'PSO':
        return PSOTorch(p, num_particles=num_agents, seed=seed), True
    if name == 'DE':
        return DETorch(p, num_particles=num_agents, seed=seed), True
    if name == 'CMA-ES':
        return CMAESTorch(p, sigma0=0.3, seed=seed), True
    if name == 'QUIMAD 1ag':
        return QIMADTorch(p, num_agents=1, eta=eta, seed=seed,
                          cooling='cosine', total_steps=epochs, min_temp=0.05), True
    if name == 'QUIMAD 4ag':
        return QIMADTorch(p, num_agents=4, eta=eta, seed=seed,
                          cooling='cosine', total_steps=epochs, min_temp=0.05), True
    if name == 'QUIMAD 8ag':
        return QIMADTorch(p, num_agents=num_agents, eta=eta, seed=seed,
                          cooling='cosine', total_steps=epochs, min_temp=0.05), True
    raise ValueError(f'Optimizador desconocido: {name}')


# ── Runner ────────────────────────────────────────────────────────────────────

def run_comparison(task, selected_opts, num_agents, eta, epochs, seed):
    if not selected_opts:
        return None, "Selecciona al menos un optimizador."

    X, y, loss_type = make_data(task, seed=seed)

    if loss_type == 'mse':
        crit = nn.MSELoss()
    else:
        crit = nn.BCELoss()

    all_curves = {}
    summary_rows = []

    for name in selected_opts:
        model = make_model(task, seed=seed)
        try:
            opt, is_q = make_optimizer(name, model, num_agents, eta, seed, epochs)
        except Exception as e:
            summary_rows.append(f"  {name}: ERROR — {e}")
            continue

        def cl():
            opt.zero_grad()
            out  = model(X)
            loss = crit(out, y)
            loss.backward()
            return loss

        curve = []
        t0 = time.perf_counter()
        for _ in range(epochs):
            if is_q:
                val = opt.step(cl)
            else:
                opt.zero_grad()
                out  = model(X)
                loss = crit(out, y)
                loss.backward()
                opt.step()
                val = loss.item()
            curve.append(float(val))
        elapsed = time.perf_counter() - t0

        all_curves[name] = curve
        summary_rows.append(
            f"  {name:<18}  loss final={curve[-1]:.5f}   tiempo={elapsed:.2f}s"
        )

    # ── Plot ──────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ep = np.arange(1, epochs + 1)

    for name, curve in all_curves.items():
        c = COLORS.get(name, '#333333')
        ax.semilogy(ep, curve, color=c, lw=2.5, label=name)

    ax.set_title(f'Convergencia — {task}', fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    ax.set_ylabel('Loss (escala log)')
    ax.legend(fontsize=9, loc='upper right')
    ax.grid(True, alpha=0.3)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    fig.text(0.5, -0.02,
        'Autor: Leonardo Jimenez Martinez — Centro de Biomatematicas BIOMAT',
        ha='center', fontsize=7.5, color='#777777', style='italic')
    plt.tight_layout()
    return fig, "\n".join(summary_rows)


# ── Interfaz Gradio ───────────────────────────────────────────────────────────

DESCRIPTION = """
# QUIMAD — Optimizador cuántico-inspirado interactivo

**Autor: Leonardo Jiménez Martínez — Centro de Biomatemáticas BIOMAT**

Compara **QIMADTorch** contra Adam, SGD, PSO, DE y CMA-ES en tiempo real.
Elige la tarea, los optimizadores y los hiperparámetros, y observa cómo
cada algoritmo baja el loss en distintos paisajes de optimización.

> QUIMAD combina **RMSProp adaptivo** + **comunicación por fidelidad cuántica** +
> **túnel cuántico** con **cooling schedule** coseno.
"""

with gr.Blocks(title="QUIMAD Space") as demo:
    gr.Markdown(DESCRIPTION)

    with gr.Row():
        with gr.Column(scale=1):
            task = gr.Dropdown(
                label="Tarea",
                choices=['Convexa (regresion lineal)',
                         'Multimodal (paisaje con valles)',
                         'Clasificacion binaria'],
                value='Multimodal (paisaje con valles)',
            )
            selected_opts = gr.CheckboxGroup(
                label="Optimizadores a comparar",
                choices=list(COLORS.keys()),
                value=['Adam', 'QUIMAD 8ag', 'PSO'],
            )
            with gr.Row():
                num_agents = gr.Slider(2, 16, value=8, step=1,
                                       label="Agentes / partículas (QUIMAD, PSO, DE)")
                eta = gr.Slider(1e-4, 0.2, value=0.01, step=1e-4,
                                label="Tasa de aprendizaje eta")
            with gr.Row():
                epochs = gr.Slider(10, 300, value=100, step=10, label="Epochs")
                seed   = gr.Slider(0, 99, value=42, step=1, label="Semilla")

            run_btn = gr.Button("Ejecutar comparación", variant="primary")

        with gr.Column(scale=2):
            plot_out    = gr.Plot(label="Curvas de convergencia")
            summary_out = gr.Textbox(label="Resumen de resultados",
                                     lines=10, interactive=False)

    run_btn.click(
        fn=run_comparison,
        inputs=[task, selected_opts, num_agents, eta, epochs, seed],
        outputs=[plot_out, summary_out],
    )

    gr.Markdown("""
---
### Guía rápida
| Parámetro | Efecto |
|---|---|
| **Agentes** | Tamaño del enjambre para QUIMAD, PSO y DE. Más agentes = más exploración pero más cómputo |
| **eta** | Tasa de aprendizaje base. Valores típicos: 0.001–0.05 |
| **Epochs** | Número de iteraciones. Los métodos sin gradiente (PSO, DE, CMA-ES) necesitan más |
| **Semilla** | Controla la aleatoriedad. Misma semilla = resultados reproducibles |

> **Nota:** PSO, DE y CMA-ES son métodos **sin gradiente** — llaman al modelo N veces por epoch
> sin usar backpropagation. Por eso son más lentos por epoch que Adam/SGD/QUIMAD.
""")

if __name__ == '__main__':
    demo.launch()
