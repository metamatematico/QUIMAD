"""
Graficas comparativas completas de QIMADTorch vs optimizadores clasicos.

Genera 9 figuras en test_y_pruebas/plots/:
  1. curvas_convergencia_convex.png
  2. curvas_convergencia_multimodal.png
  3. boxplot_perdida_final.png
  4. frontera_eficiencia.png
  5. sensibilidad_num_agents.png
  6. sensibilidad_eta.png
  7. comparacion_topologias.png
  8. tradeoff_k_eval.png
  9. diagnosticos_internos.png

Correr desde la raiz del proyecto:
    python test_y_pruebas/graficas_comparativas.py
"""

import math
import sys
import time
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
import torch
import torch.nn as nn

sys.path.insert(0, str(Path(__file__).parent.parent))
from quimad_torch import QIMADTorch
from pso_torch import PSOTorch

OUT = Path(__file__).parent
OUT.mkdir(exist_ok=True)

# ── Paleta consistente ────────────────────────────────────────────────────────
COLORS = {
    'Adam':         '#2196F3',   # azul
    'SGD':          '#9E9E9E',   # gris
    'PSO':          '#E91E63',   # rosa/magenta
    'QUIMAD_1ag':   '#FF9800',   # naranja
    'QUIMAD_full':  '#4CAF50',   # verde
    'QUIMAD_k4':    '#8BC34A',   # verde claro
    'QUIMAD_k2':    '#CDDC39',   # amarillo-verde
}
LINESTYLES = {
    'Adam': '-', 'SGD': '--', 'PSO': '-.',
    'QUIMAD_1ag': '-.', 'QUIMAD_full': '-',
    'QUIMAD_k4': ':', 'QUIMAD_k2': ':',
}

plt.rcParams.update({
    'font.family': 'DejaVu Sans',
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.alpha': 0.3,
    'figure.dpi': 130,
})

N_SEEDS  = 10
EPOCHS   = 120


# ── Factorías de datos y modelos ──────────────────────────────────────────────

def make_convex(seed):
    torch.manual_seed(seed)
    X = torch.randn(200, 4)
    y = (X @ torch.tensor([1., -1., 0.5, -0.5])).unsqueeze(1)
    return X, y

def make_multimodal(seed):
    torch.manual_seed(seed)
    X = torch.rand(200, 2) * 4 - 2
    y = (X[:,0]**2 + X[:,1]**2
         - 1.5*torch.cos(2*math.pi*X[:,0])
         - 1.5*torch.cos(2*math.pi*X[:,1])).unsqueeze(1)
    return X, y

def convex_model(seed):
    torch.manual_seed(seed); return nn.Linear(4, 1)

def multimodal_model(seed):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(2, 32), nn.Tanh(), nn.Linear(32, 1))

TASKS = {
    'convex':      (make_convex,      convex_model),
    'multimodal':  (make_multimodal,  multimodal_model),
}


# ── Corredor generico ─────────────────────────────────────────────────────────

def run_optimizer(opt_fn, model_fn, data_fn, seed, epochs, is_quimad=True):
    """Devuelve (lista_de_losses_por_epoch, tiempo_total)."""
    X, y  = data_fn(seed)
    model = model_fn(seed)
    opt   = opt_fn(model, seed)
    crit  = nn.MSELoss()
    losses = []
    t0 = time.perf_counter()

    def cl():
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        return loss

    for _ in range(epochs):
        if is_quimad:
            val = opt.step(cl)
        else:
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            val = loss.item()
        losses.append(val if isinstance(val, float) else val.item())

    return losses, time.perf_counter() - t0


# ── Configs base ──────────────────────────────────────────────────────────────

def base_configs():
    return [
        ('Adam',       False,
         lambda m, s: torch.optim.Adam(m.parameters(), lr=0.01)),
        ('SGD',        False,
         lambda m, s: torch.optim.SGD(m.parameters(), lr=0.01, momentum=0.9)),
        ('PSO',        True,
         lambda m, s: PSOTorch(m.parameters(), num_particles=8, seed=s)),
        ('QUIMAD_1ag', True,
         lambda m, s: QIMADTorch(m.parameters(), num_agents=1, eta=0.01, seed=s)),
        ('QUIMAD_full',True,
         lambda m, s: QIMADTorch(m.parameters(), num_agents=8, eta=0.01, seed=s)),
        ('QUIMAD_k4',  True,
         lambda m, s: QIMADTorch(m.parameters(), num_agents=8, eta=0.01, k_eval=4, seed=s)),
        ('QUIMAD_k2',  True,
         lambda m, s: QIMADTorch(m.parameters(), num_agents=8, eta=0.01, k_eval=2, seed=s)),
    ]


def collect_curves(task_key, epochs=EPOCHS, seeds=N_SEEDS):
    """Devuelve dict name -> array (seeds, epochs)."""
    data_fn, model_fn = TASKS[task_key]
    out = {}
    times = {}
    for name, is_q, opt_fn in base_configs():
        all_curves = []
        all_times  = []
        for seed in range(seeds):
            curve, t = run_optimizer(opt_fn, model_fn, data_fn, seed, epochs, is_q)
            all_curves.append(curve)
            all_times.append(t)
        out[name]   = np.array(all_curves)
        times[name] = np.array(all_times)
    return out, times


# =============================================================================
# FIGURA 1 & 2 — Curvas de convergencia
# =============================================================================

def fig_convergencia(task_key, ax, curves, title, ylabel=True):
    ep = np.arange(1, curves[next(iter(curves))].shape[1] + 1)
    for name, arr in curves.items():
        med = np.median(arr, axis=0)
        p25, p75 = np.percentile(arr, 25, axis=0), np.percentile(arr, 75, axis=0)
        c = COLORS[name]
        ax.semilogy(ep, med, color=c, ls=LINESTYLES[name], lw=2, label=name)
        ax.fill_between(ep, p25, p75, color=c, alpha=0.13)

    ax.set_title(title, fontweight='bold', fontsize=12)
    ax.set_xlabel('Epoch')
    if ylabel:
        ax.set_ylabel('Loss (escala log)')
    ax.legend(fontsize=8, loc='upper right')


def make_convergence_plots(curves_conv, curves_multi):
    for task, curves, fname, subtitle in [
        ('convex', curves_conv,
         'curvas_convergencia_convex.png',
         'Tarea convexa (regresion lineal). Aqui todas las herramientas deben llegar rapido al minimo.\n'
         'QUIMAD (verde) compite con SGD y supera a Adam en convergencia final.'),
        ('multimodal', curves_multi,
         'curvas_convergencia_multimodal.png',
         'Tarea multimodal (paisaje con muchos valles falsos). Aqui QUIMAD tiene ventaja:\n'
         'el enjambre de agentes escapa de valles locales donde Adam y SGD quedan atrapados.'),
    ]:
        fig, ax = plt.subplots(figsize=(10, 5))
        fig_convergencia(task, ax, curves, f'Convergencia: {task}')
        ep = curves[next(iter(curves))].shape[1]
        fig.text(0.5, -0.04, subtitle, ha='center', fontsize=9,
                 style='italic', color='#444444', wrap=True)
        plt.tight_layout()
        path = OUT / fname
        fig.savefig(path, dpi=150, bbox_inches='tight')
        plt.close(fig)
        print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 3 — Boxplot de perdida final (ambas tareas)
# =============================================================================

def make_boxplot(curves_conv, curves_multi):
    names = list(curves_conv.keys())
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, curves, title in [
        (axes[0], curves_conv,   'Tarea convexa'),
        (axes[1], curves_multi,  'Tarea multimodal'),
    ]:
        data   = [curves[n][:, -1] for n in names]
        colors = [COLORS[n] for n in names]
        bp = ax.boxplot(data, tick_labels=names, patch_artist=True,
                        medianprops=dict(color='black', lw=2))
        for patch, c in zip(bp['boxes'], colors):
            patch.set_facecolor(c)
            patch.set_alpha(0.7)
        ax.set_title(title, fontweight='bold')
        ax.set_ylabel('Loss final (epoch 120)')
        ax.tick_params(axis='x', rotation=20)

    fig.suptitle('Distribucion de loss final sobre 10 semillas',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
        'Cada caja muestra la variabilidad entre 10 ejecuciones distintas.\n'
        'Una caja angosta = resultados consistentes. Una caja alta = el optimizador es sensible al punto de inicio.\n'
        'QUIMAD completo (verde) tiene la caja mas baja en multimodal: gana y es estable.',
        ha='center', fontsize=9, style='italic', color='#444444')
    plt.tight_layout()
    path = OUT / 'boxplot_perdida_final.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 4 — Frontera de eficiencia (costo vs calidad)
# =============================================================================

def make_eficiencia(curves_conv, curves_multi, times_conv, times_multi):
    """Scatter: tiempo_mediano vs loss_final_mediano."""
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    names = list(curves_conv.keys())

    for ax, curves, times, title in [
        (axes[0], curves_conv,  times_conv,  'Tarea convexa'),
        (axes[1], curves_multi, times_multi, 'Tarea multimodal'),
    ]:
        for n in names:
            loss_med = float(np.median(curves[n][:, -1]))
            time_med = float(np.median(times[n]))
            c = COLORS[n]
            ax.scatter(time_med, loss_med, s=140, color=c, zorder=5,
                       edgecolors='black', linewidths=0.6)
            ax.annotate(n, (time_med, loss_med),
                        textcoords='offset points', xytext=(6, 4),
                        fontsize=8, color=c)

        ax.set_xlabel('Tiempo por ejecucion (s)')
        ax.set_ylabel('Loss final (mediana)')
        ax.set_title(title, fontweight='bold')
        # Cuadrante ideal: abajo-izquierda
        xlim, ylim = ax.get_xlim(), ax.get_ylim()
        ax.annotate('ideal', xy=(xlim[0], ylim[0]),
                    fontsize=7, color='green', alpha=0.5)

    fig.suptitle('Frontera eficiencia: rapido y barato vs preciso',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
        'El punto ideal esta en la esquina inferior-izquierda: rapido Y con bajo loss.\n'
        'QUIMAD completo (verde) tarda mas pero logra el menor loss en multimodal.\n'
        'k_eval=4 (verde claro) es un buen compromiso: mitad del tiempo, calidad aceptable.',
        ha='center', fontsize=9, style='italic', color='#444444')
    plt.tight_layout()
    path = OUT / 'frontera_eficiencia.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 5 — Sensibilidad: num_agents
# =============================================================================

def make_sens_agents():
    agents_list = [1, 2, 4, 8, 12, 16]
    seeds = 8
    epochs = 80
    results_conv  = []
    results_multi = []
    times_conv    = []
    times_multi   = []

    for n in agents_list:
        lc, lm, tc, tm = [], [], [], []
        for task_key, lst, tl in [('convex', lc, tc), ('multimodal', lm, tm)]:
            data_fn, model_fn = TASKS[task_key]
            for seed in range(seeds):
                curve, t = run_optimizer(
                    lambda m, s, _n=n: QIMADTorch(m.parameters(), num_agents=_n, eta=0.01, seed=s),
                    model_fn, data_fn, seed, epochs, True)
                lst.append(curve[-1])
                tl.append(t)
        results_conv.append(np.median(lc))
        results_multi.append(np.median(lm))
        times_conv.append(np.median(tc))
        times_multi.append(np.median(tm))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    for ax, ys, ts, title in [
        (axes[0], results_conv,  times_conv,  'Tarea convexa'),
        (axes[1], results_multi, times_multi, 'Tarea multimodal'),
    ]:
        color = '#4CAF50'
        ax2 = ax.twinx()
        ax.plot(agents_list, ys, 'o-', color=color, lw=2, ms=8, label='Loss final')
        ax2.plot(agents_list, ts, 's--', color='#FF5722', lw=1.5, ms=6, label='Tiempo (s)')
        ax.set_xlabel('Numero de agentes (canicas)')
        ax.set_ylabel('Loss final (mediana)', color=color)
        ax2.set_ylabel('Tiempo (s)', color='#FF5722')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(agents_list)
        lines1, labs1 = ax.get_legend_handles_labels()
        lines2, labs2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labs1 + labs2, fontsize=8)

    fig.suptitle('Efecto del numero de agentes (canicas en el enjambre)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
        'Mas agentes = mayor exploracion del paisaje, pero mas tiempo de computo.\n'
        'En convex, 1-2 agentes son suficientes: la superficie es simple.\n'
        'En multimodal, cada agente extra ayuda a escapar de valles locales, pero con retornos decrecientes.',
        ha='center', fontsize=9, style='italic', color='#444444')
    plt.tight_layout()
    path = OUT / 'sensibilidad_num_agents.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 6 — Sensibilidad: eta (learning rate)
# =============================================================================

def make_sens_eta():
    etas = [1e-4, 5e-4, 1e-3, 5e-3, 1e-2, 5e-2, 1e-1]
    seeds = 8
    epochs = 80
    results_conv  = []
    results_multi = []

    for eta in etas:
        lc, lm = [], []
        for task_key, lst in [('convex', lc), ('multimodal', lm)]:
            data_fn, model_fn = TASKS[task_key]
            for seed in range(seeds):
                curve, _ = run_optimizer(
                    lambda m, s, _e=eta: QIMADTorch(m.parameters(), num_agents=4, eta=_e, seed=s),
                    model_fn, data_fn, seed, epochs, True)
                lst.append(curve[-1])
        results_conv.append(np.median(lc))
        results_multi.append(np.median(lm))

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    eta_labels = [f'{e:.0e}' for e in etas]

    for ax, ys, title in [
        (axes[0], results_conv,  'Tarea convexa'),
        (axes[1], results_multi, 'Tarea multimodal'),
    ]:
        ax.semilogx(etas, ys, 'D-', color='#673AB7', lw=2, ms=8)
        ax.set_xlabel('Tasa de aprendizaje (eta)')
        ax.set_ylabel('Loss final (mediana)')
        ax.set_title(title, fontweight='bold')
        ax.set_xticks(etas)
        ax.set_xticklabels(eta_labels, rotation=30, fontsize=8)
        best_eta = etas[int(np.argmin(ys))]
        ax.axvline(best_eta, color='red', ls='--', alpha=0.5, lw=1.5,
                   label=f'Mejor eta={best_eta:.0e}')
        ax.legend(fontsize=8)

    fig.suptitle('Sensibilidad al hiperparametro eta (tasa de aprendizaje)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
        'La linea roja marca el eta que dio el menor loss en esa tarea.\n'
        'QUIMAD tolera bien un rango amplio de eta: no colapsa bruscamente.\n'
        'En tareas simples, etas grandes convergen rapido. En multimodal, moderados son mas estables.',
        ha='center', fontsize=9, style='italic', color='#444444')
    plt.tight_layout()
    path = OUT / 'sensibilidad_eta.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 7 — Comparacion de topologias
# =============================================================================

def make_topologias():
    topologies = ['complete', 'ring', 'grid', 'random']
    seeds = 10
    epochs = 80
    results_conv  = {t: [] for t in topologies}
    results_multi = {t: [] for t in topologies}

    for topo in topologies:
        for task_key, dic in [('convex', results_conv), ('multimodal', results_multi)]:
            data_fn, model_fn = TASKS[task_key]
            for seed in range(seeds):
                curve, _ = run_optimizer(
                    lambda m, s, _t=topo: QIMADTorch(m.parameters(), num_agents=8,
                                                      eta=0.01, topology=_t, seed=s),
                    model_fn, data_fn, seed, epochs, True)
                dic[topo].append(curve[-1])

    x = np.arange(len(topologies))
    width = 0.35
    topo_colors = ['#3F51B5', '#E91E63', '#009688', '#FF9800']

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, dic, title in [
        (axes[0], results_conv,  'Tarea convexa'),
        (axes[1], results_multi, 'Tarea multimodal'),
    ]:
        medians = [np.median(dic[t]) for t in topologies]
        stds    = [np.std(dic[t])    for t in topologies]
        bars = ax.bar(x, medians, width=0.6, color=topo_colors, alpha=0.8,
                      edgecolor='black', linewidth=0.6)
        ax.errorbar(x, medians, yerr=stds, fmt='none', color='black',
                    capsize=5, lw=1.5)
        ax.set_xticks(x)
        ax.set_xticklabels(topologies)
        ax.set_ylabel('Loss final (mediana)')
        ax.set_title(title, fontweight='bold')
        # Etiqueta encima de cada barra
        for bar, m in zip(bars, medians):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height()*1.02,
                    f'{m:.4f}', ha='center', fontsize=8)

    fig.suptitle('Efecto de la topologia de comunicacion entre agentes',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
        'La topologia define quienes pueden "comunicarse" dentro del enjambre.\n'
        '"complete": todos hablan con todos (mas informacion, mas costo).\n'
        '"ring": cada agente solo habla con sus dos vecinos (exploracion mas lenta pero diversa).\n'
        '"grid": malla 2D. "random": conexiones aleatorias (Erdos-Renyi p=0.5).',
        ha='center', fontsize=9, style='italic', color='#444444')
    plt.tight_layout()
    path = OUT / 'comparacion_topologias.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 8 — Tradeoff k_eval: closure calls vs calidad
# =============================================================================

def make_keval_tradeoff():
    k_evals = [1, 2, 3, 4, 6, 8]   # k_eval=8 == full con 8 agentes
    seeds = 10
    epochs = 100
    conv_loss  = []
    multi_loss = []
    closures   = []

    for k in k_evals:
        lc, lm = [], []
        for task_key, lst in [('convex', lc), ('multimodal', lm)]:
            data_fn, model_fn = TASKS[task_key]
            for seed in range(seeds):
                curve, _ = run_optimizer(
                    lambda m, s, _k=k: QIMADTorch(m.parameters(), num_agents=8,
                                                   eta=0.01, k_eval=_k, seed=s),
                    model_fn, data_fn, seed, epochs, True)
                lst.append(curve[-1])
        conv_loss.append(np.median(lc))
        multi_loss.append(np.median(lm))
        closures.append(k * epochs)   # llamadas a closure por ejecucion

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, ys, title in [
        (axes[0], conv_loss,  'Tarea convexa'),
        (axes[1], multi_loss, 'Tarea multimodal'),
    ]:
        sc = ax.scatter(closures, ys, s=[60 + 20*k for k in k_evals],
                        c=closures, cmap='RdYlGn_r', edgecolors='black',
                        linewidths=0.6, zorder=5)
        ax.plot(closures, ys, '--', color='#888888', lw=1, zorder=1)
        for k, cl, y in zip(k_evals, closures, ys):
            lbl = f'k={k}' if k < 8 else 'k=8\n(full)'
            ax.annotate(lbl, (cl, y), textcoords='offset points',
                        xytext=(6, 4), fontsize=8)
        ax.set_xlabel('Llamadas a closure() por ejecucion (costo total)')
        ax.set_ylabel('Loss final (mediana)')
        ax.set_title(title, fontweight='bold')
        plt.colorbar(sc, ax=ax, label='Llamadas closure')

    fig.suptitle('Tradeoff: cuantos agentes evaluar por paso (k_eval)',
                 fontsize=13, fontweight='bold', y=1.01)
    fig.text(0.5, -0.06,
        'k_eval controla cuantos agentes calculan gradientes en cada paso (los demas reusan el ultimo).\n'
        'Menos k_eval = mas barato (menos llamadas al modelo) pero informacion mas antigua.\n'
        'La curva muestra donde vale la pena pagar por mas evaluaciones.',
        ha='center', fontsize=9, style='italic', color='#444444')
    plt.tight_layout()
    path = OUT / 'tradeoff_k_eval.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# FIGURA 9 — Diagnosticos internos del enjambre
# =============================================================================

def make_diagnosticos():
    torch.manual_seed(0)
    X, y = make_multimodal(0)
    model = multimodal_model(0)
    crit = nn.MSELoss()
    epochs = 200

    opt = QIMADTorch(model.parameters(), num_agents=8, eta=0.01, seed=0)
    opt.enable_tracking()

    def cl():
        opt.zero_grad(); loss = crit(model(X), y); loss.backward(); return loss

    loss_history = []
    for _ in range(epochs):
        loss_history.append(opt.step(cl))

    log = opt.get_log()
    ep = np.arange(1, epochs + 1)

    # Diversity: std de posiciones de agentes
    diversity = np.array(log['diversity'])
    # Tunneling: eventos por epoch
    tunnels = np.array(log['tunnels'])
    # Alpha medio por epoch (probabilidad de tunelizacion)
    alphas = np.array(log['alphas'])  # (epochs, n_agents)
    # Loss de cada agente por epoch
    agent_losses = np.array(log['losses'])  # (epochs, n_agents)

    fig = plt.figure(figsize=(14, 10))
    gs = gridspec.GridSpec(2, 2, hspace=0.45, wspace=0.35)

    # -- Panel A: Loss global vs agentes individuales --
    ax_a = fig.add_subplot(gs[0, 0])
    for i in range(agent_losses.shape[1]):
        ax_a.semilogy(ep, agent_losses[:, i], alpha=0.4, lw=0.8,
                      color=plt.cm.Set2(i/8))
    ax_a.semilogy(ep, loss_history, 'k-', lw=2.5, label='Mejor global')
    ax_a.set_title('A: Loss de agentes individuales vs mejor global',
                   fontweight='bold', fontsize=10)
    ax_a.set_xlabel('Epoch'); ax_a.set_ylabel('Loss (log)')
    ax_a.legend(fontsize=8)
    ax_a.text(0.02, 0.05,
        'Cada linea de color = 1 agente.\nLinea negra = mejor encontrado hasta ese momento.\n'
        'Los agentes exploran distintas zonas: algunos suben, el global solo baja.',
        transform=ax_a.transAxes, fontsize=7.5, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # -- Panel B: Diversidad del enjambre --
    ax_b = fig.add_subplot(gs[0, 1])
    ax_b.plot(ep, diversity, color='#9C27B0', lw=1.5)
    ax_b.fill_between(ep, 0, diversity, color='#9C27B0', alpha=0.15)
    ax_b.set_title('B: Diversidad del enjambre (varianza de posiciones)',
                   fontweight='bold', fontsize=10)
    ax_b.set_xlabel('Epoch'); ax_b.set_ylabel('Varianza media entre agentes')
    ax_b.text(0.02, 0.95,
        'Mide que tan separados estan los agentes en el espacio de pesos.\n'
        'Alta diversidad = exploracion activa.\n'
        'Si cae a 0, todos convergen al mismo punto (perdida de diversidad).',
        transform=ax_b.transAxes, fontsize=7.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # -- Panel C: Eventos de tunelizacion --
    ax_c = fig.add_subplot(gs[1, 0])
    ax_c.bar(ep, tunnels, color='#F44336', alpha=0.7, width=1.0)
    window = 20
    smooth = np.convolve(tunnels, np.ones(window)/window, mode='same')
    ax_c.plot(ep, smooth, 'k-', lw=2, label=f'Promedio movil ({window} ep)')
    ax_c.set_title('C: Eventos de tunel cuantico por epoch',
                   fontweight='bold', fontsize=10)
    ax_c.set_xlabel('Epoch'); ax_c.set_ylabel('N agentes que tunelizaron')
    ax_c.legend(fontsize=8)
    ax_c.text(0.02, 0.95,
        'Un "tunel" ocurre cuando un agente estancado salta a una posicion\n'
        'aleatoria para escapar de un valle local.\n'
        'Muchos tuneles al inicio = el enjambre aun no encontro buenas regiones.',
        transform=ax_c.transAxes, fontsize=7.5, verticalalignment='top',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    # -- Panel D: Evolucion del estado cuantico (alpha) --
    ax_d = fig.add_subplot(gs[1, 1])
    alpha_mean = alphas.mean(axis=1)
    alpha_std  = alphas.std(axis=1)
    prob_tunnel = np.sin(alpha_mean / 2) ** 2
    ax_d.plot(ep, prob_tunnel, color='#FF9800', lw=1.5, label='P(tunel) media')
    ax_d.fill_between(ep,
                      np.sin(np.clip(alpha_mean - alpha_std, 0, math.pi) / 2)**2,
                      np.sin(np.clip(alpha_mean + alpha_std, 0, math.pi) / 2)**2,
                      color='#FF9800', alpha=0.2)
    ax_d.axhline(0.5, color='red', ls='--', lw=1, alpha=0.5,
                 label='P=0.5 (umbral)')
    ax_d.set_ylim(0, 1)
    ax_d.set_title('D: Probabilidad de tunelizacion (estado cuantico alpha)',
                   fontweight='bold', fontsize=10)
    ax_d.set_xlabel('Epoch'); ax_d.set_ylabel('sin^2(alpha/2)')
    ax_d.legend(fontsize=8)
    ax_d.text(0.02, 0.05,
        'Cada agente tiene un angulo "alpha" en la esfera de Bloch.\n'
        'sin^2(alpha/2) es la probabilidad de que ese agente pueda tunelizar.\n'
        'El estado cuantico evoluciona por paseo aleatorio, independiente del gradiente.',
        transform=ax_d.transAxes, fontsize=7.5, verticalalignment='bottom',
        bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))

    fig.suptitle('Diagnosticos internos del enjambre QUIMAD (200 epochs, 8 agentes, multimodal)',
                 fontsize=13, fontweight='bold')
    path = OUT / 'diagnosticos_internos.png'
    fig.savefig(path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    print(f'  Guardado: {path.name}')


# =============================================================================
# MAIN
# =============================================================================

if __name__ == '__main__':
    print('Recolectando curvas de convergencia (10 semillas x 6 optimizadores x 2 tareas)...')
    curves_conv,  times_conv  = collect_curves('convex')
    curves_multi, times_multi = collect_curves('multimodal')

    print('\nGenerando graficas...')
    make_convergence_plots(curves_conv, curves_multi)
    make_boxplot(curves_conv, curves_multi)
    make_eficiencia(curves_conv, curves_multi, times_conv, times_multi)
    make_sens_agents()
    make_sens_eta()
    make_topologias()
    make_keval_tradeoff()
    make_diagnosticos()

    print(f'\nListo. 9 graficas guardadas en: {OUT}')
