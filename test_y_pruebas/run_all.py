"""
Bateria completa de pruebas para QIMADTorch.

Secciones:
  1. Tests unitarios (pytest)
  2. Benchmark comparativo: QUIMAD vs Adam vs SGD (10 seeds x 2 tareas)
  3. Sensibilidad de hiperparametros (num_agents, eta, topology)
  4. Diagnosticos detallados (1 corrida, 200 epochs, tracking completo)
  5. Genera RESULTADOS.md y plots/

Correr desde la raiz del proyecto:
    python test_y_pruebas/run_all.py

Tiempo estimado: 10-15 minutos.
"""
import math
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT))

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
import torch
import torch.nn as nn

from quimad_torch import QIMADTorch

OUTDIR = Path(__file__).parent
PLOTS = OUTDIR / 'plots'
PLOTS.mkdir(exist_ok=True)


# ══════════════════════════════════════════════════════════════════════════════
# Utilidades compartidas
# ══════════════════════════════════════════════════════════════════════════════

def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(
        nn.Linear(2, 64), nn.Tanh(),
        nn.Linear(64, 32), nn.Tanh(),
        nn.Linear(32, 1),
    )

def make_multimodal_data():
    torch.manual_seed(0)
    N = 300
    X = torch.rand(N, 2) * 4 - 2
    y = (X[:, 0] ** 2 + X[:, 1] ** 2
         - 1.5 * torch.cos(2 * math.pi * X[:, 0])
         - 1.5 * torch.cos(2 * math.pi * X[:, 1])).unsqueeze(1)
    y += 0.05 * torch.randn_like(y)
    return X, y

def make_convex_data():
    torch.manual_seed(0)
    N = 300
    X = torch.randn(N, 2)
    y = (X[:, 0] + 2.0 * X[:, 1]).unsqueeze(1) + 0.05 * torch.randn(N, 1)
    return X, y

def train_run(opt_name, X, y, seed, epochs, **quimad_kwargs):
    torch.manual_seed(seed)
    model = make_model(seed)
    crit = nn.MSELoss()

    if opt_name == 'QIMADTorch':
        kw = dict(num_agents=8, eta=0.01, seed=seed)
        kw.update(quimad_kwargs)
        opt = QIMADTorch(model.parameters(), **kw)
        def closure():
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            return loss
        t0 = time.perf_counter()
        curve = [opt.step(closure) for _ in range(epochs)]
        elapsed = time.perf_counter() - t0

    else:
        if opt_name == 'Adam':
            opt = torch.optim.Adam(model.parameters(), lr=0.01)
        else:
            opt = torch.optim.SGD(model.parameters(), lr=0.01, momentum=0.9)
        t0 = time.perf_counter()
        curve = []
        for _ in range(epochs):
            opt.zero_grad()
            loss = crit(model(X), y)
            loss.backward()
            opt.step()
            curve.append(loss.item())
        elapsed = time.perf_counter() - t0

    return curve, elapsed


# ══════════════════════════════════════════════════════════════════════════════
# Seccion 1: Tests unitarios
# ══════════════════════════════════════════════════════════════════════════════

def run_unit_tests():
    print("\n[1/4] Tests unitarios (pytest)...")
    result = subprocess.run(
        [sys.executable, '-m', 'pytest', str(OUTDIR / 'test_unit.py'), '-v', '--tb=short'],
        capture_output=True, text=True, cwd=str(ROOT)
    )
    output = result.stdout + result.stderr
    # Parse summary line
    lines = output.strip().splitlines()
    summary = next((l for l in reversed(lines) if 'passed' in l or 'failed' in l or 'error' in l), 'sin resultados')
    print(f"   {summary}")
    return output, summary


# ══════════════════════════════════════════════════════════════════════════════
# Seccion 2: Benchmark comparativo
# ══════════════════════════════════════════════════════════════════════════════

def run_benchmark(n_seeds=10, epochs=100):
    print(f"\n[2/4] Benchmark comparativo ({n_seeds} seeds x 3 optimizadores x 2 tareas)...")
    tasks = {
        'Convexa': make_convex_data(),
        'Multimodal': make_multimodal_data(),
    }
    optimizers = ['QIMADTorch', 'Adam', 'SGD']
    records = []
    all_curves = {task: {opt: [] for opt in optimizers} for task in tasks}

    for task_name, (X, y) in tasks.items():
        for opt_name in optimizers:
            for seed in range(n_seeds):
                curve, elapsed = train_run(opt_name, X, y, seed=42 + seed, epochs=epochs)
                records.append({
                    'task': task_name, 'optimizer': opt_name,
                    'seed': seed, 'final_loss': curve[-1],
                    'initial_loss': curve[0], 'elapsed_s': elapsed,
                })
                all_curves[task_name][opt_name].append(curve)
                print(f"   {task_name:12s} {opt_name:12s} seed={seed}  loss={curve[-1]:.5f}")

    df = pd.DataFrame(records)
    df.to_csv(OUTDIR / 'benchmark_results.csv', index=False)

    # Descriptive stats
    stats_df = (
        df.groupby(['task', 'optimizer'])['final_loss']
        .agg(mean='mean', std='std', median='median', best='min', worst='max')
        .round(5)
        .reset_index()
    )
    stats_df['mean_std'] = (
        stats_df['mean'].map('{:.4f}'.format) + ' +/- ' + stats_df['std'].map('{:.4f}'.format)
    )

    # Wilcoxon tests
    wilcoxon_rows = []
    for task in df['task'].unique():
        q_vals = df[(df['task'] == task) & (df['optimizer'] == 'QIMADTorch')]['final_loss'].values
        for baseline in ['Adam', 'SGD']:
            b_vals = df[(df['task'] == task) & (df['optimizer'] == baseline)]['final_loss'].values
            n = min(len(q_vals), len(b_vals))
            try:
                _, p = stats.wilcoxon(q_vals[:n], b_vals[:n], alternative='two-sided')
                winner = 'QIMADTorch' if (p < 0.05 and np.median(q_vals) < np.median(b_vals)) else \
                         (baseline if (p < 0.05 and np.median(q_vals) > np.median(b_vals)) else 'Sin diferencia')
            except ValueError:
                p, winner = 1.0, 'Empate'
            wilcoxon_rows.append({'task': task, 'vs': baseline, 'p_value': round(p, 5), 'winner': winner})
    wilcoxon_df = pd.DataFrame(wilcoxon_rows)

    # Time comparison
    time_df = df.groupby('optimizer')['elapsed_s'].mean().round(3).reset_index()
    time_df.columns = ['optimizer', 'avg_seconds_per_run']

    # Plot convergence curves
    _plot_convergence(all_curves, tasks, epochs, n_seeds)

    return stats_df, wilcoxon_df, time_df, df


def _plot_convergence(all_curves, tasks, epochs, n_seeds):
    colors = {'QIMADTorch': '#e07b39', 'Adam': '#3a7ebf', 'SGD': '#4caf50'}
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    for ax, (task_name, _) in zip(axes, tasks.items()):
        for opt_name, curves in all_curves[task_name].items():
            arr = np.array(curves)
            mean = arr.mean(axis=0)
            std = arr.std(axis=0)
            x = np.arange(epochs)
            ax.semilogy(x, mean, label=opt_name, color=colors[opt_name], linewidth=2)
            ax.fill_between(x, np.clip(mean - std, 1e-6, None), mean + std,
                            alpha=0.18, color=colors[opt_name])
        ax.set_title(f'Convergencia: {task_name}')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('MSE Loss (log)')
        ax.legend(fontsize=9)
        ax.grid(True, alpha=0.3)
    plt.suptitle(f'QUIMAD vs Adam vs SGD  ({n_seeds} seeds, media +/- std)', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / 'convergencia_comparativa.png', dpi=150)
    plt.close()
    print("   Plot: plots/convergencia_comparativa.png")


# ══════════════════════════════════════════════════════════════════════════════
# Seccion 3: Sensibilidad de hiperparametros
# ══════════════════════════════════════════════════════════════════════════════

def run_sensitivity(n_seeds=5, epochs=80):
    print(f"\n[3/4] Sensibilidad de hiperparametros ({n_seeds} seeds por config)...")
    X, y = make_multimodal_data()
    records = []

    # num_agents
    for n in [1, 2, 4, 8, 16]:
        for seed in range(n_seeds):
            curve, _ = train_run('QIMADTorch', X, y, seed=42 + seed, epochs=epochs, num_agents=n)
            records.append({'param': 'num_agents', 'value': str(n), 'final_loss': curve[-1]})
        print(f"   num_agents={n:2d}  mean={np.mean([r['final_loss'] for r in records if r['param']=='num_agents' and r['value']==str(n)]):.4f}")

    # eta
    for eta in [1e-4, 1e-3, 5e-3, 1e-2, 5e-2]:
        for seed in range(n_seeds):
            curve, _ = train_run('QIMADTorch', X, y, seed=42 + seed, epochs=epochs, eta=eta)
            records.append({'param': 'eta', 'value': str(eta), 'final_loss': curve[-1]})
        print(f"   eta={eta:.1e}  mean={np.mean([r['final_loss'] for r in records if r['param']=='eta' and r['value']==str(eta)]):.4f}")

    # topology
    for topo in ['complete', 'ring', 'grid', 'random']:
        for seed in range(n_seeds):
            curve, _ = train_run('QIMADTorch', X, y, seed=42 + seed, epochs=epochs, topology=topo)
            records.append({'param': 'topology', 'value': topo, 'final_loss': curve[-1]})
        print(f"   topology={topo:8s}  mean={np.mean([r['final_loss'] for r in records if r['param']=='topology' and r['value']==topo]):.4f}")

    df = pd.DataFrame(records)

    # Plots
    _plot_sensitivity(df)

    # Summary table
    summary = (
        df.groupby(['param', 'value'])['final_loss']
        .agg(mean='mean', std='std')
        .round(4)
        .reset_index()
    )
    return summary


def _plot_sensitivity(df):
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))

    for ax, (param, xlabel) in zip(axes, [
        ('num_agents', 'Numero de agentes'),
        ('eta', 'Tasa de aprendizaje (eta)'),
        ('topology', 'Topologia'),
    ]):
        sub = df[df['param'] == param]
        values = sub['value'].unique().tolist()
        # Sort numerically for numeric params
        try:
            values = sorted(values, key=float)
        except ValueError:
            pass
        data = [sub[sub['value'] == v]['final_loss'].values for v in values]
        ax.boxplot(data, tick_labels=values, patch_artist=True,
                   boxprops=dict(facecolor='#e07b39', alpha=0.6))
        ax.set_title(f'Sensibilidad: {xlabel}')
        ax.set_xlabel(xlabel)
        ax.set_ylabel('MSE final')
        ax.grid(True, alpha=0.3, axis='y')
        if param == 'eta':
            ax.set_yscale('log')

    plt.suptitle('Sensibilidad de Hiperparametros (tarea multimodal)', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / 'sensibilidad_hiperparametros.png', dpi=150)
    plt.close()
    print("   Plot: plots/sensibilidad_hiperparametros.png")


# ══════════════════════════════════════════════════════════════════════════════
# Seccion 4: Diagnosticos detallados
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(epochs=200):
    print(f"\n[4/4] Diagnosticos detallados (1 corrida, {epochs} epochs)...")
    X, y = make_multimodal_data()
    torch.manual_seed(42)
    model = make_model(42)
    crit = nn.MSELoss()
    opt = QIMADTorch(model.parameters(), num_agents=8, eta=0.01, seed=42)
    opt.enable_tracking()

    best_curve = []
    all_agent_losses = []

    def closure():
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        return loss

    for ep in range(epochs):
        loss = opt.step(closure)
        best_curve.append(loss)
        all_agent_losses.append(opt.get_agent_losses())

    log = opt.get_log()

    # Build stats
    all_agent_arr = np.array(all_agent_losses)  # (epochs, 8)
    total_tunnels = sum(log['tunnels'])
    tunnel_epochs = [i for i, t in enumerate(log['tunnels']) if t > 0]
    diversity_arr = np.array(log['diversity'])
    final_alphas = log['alphas'][-1]
    avg_alpha_initial = np.mean(log['alphas'][0])
    avg_alpha_final = np.mean(final_alphas)

    diag_stats = {
        'total_tunnels': total_tunnels,
        'tunnel_per_epoch': round(total_tunnels / epochs, 2),
        'avg_alpha_initial': round(avg_alpha_initial, 4),
        'avg_alpha_final': round(avg_alpha_final, 4),
        'final_diversity': round(float(diversity_arr[-1]), 6) if len(diversity_arr) > 0 else 0,
        'best_loss': round(best_curve[-1], 6),
    }

    # Plots
    _plot_diagnostics(best_curve, all_agent_arr, log, epochs)

    print(f"   Tunelamiento: {total_tunnels} eventos, {diag_stats['tunnel_per_epoch']} agentes/epoch")
    print(f"   Alpha promedio: {avg_alpha_initial:.3f} (inicial) -> {avg_alpha_final:.3f} (final)")
    print(f"   Diversidad final (varianza de pesos): {diag_stats['final_diversity']:.6f}")
    print(f"   Mejor loss: {diag_stats['best_loss']:.5f}")

    return diag_stats


def _plot_diagnostics(best_curve, all_agent_arr, log, epochs):
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    epochs_x = np.arange(epochs)
    cmap = plt.cm.tab10

    # Panel 1: Trayectorias de todos los agentes + mejor global
    ax = axes[0, 0]
    for i in range(all_agent_arr.shape[1]):
        ax.semilogy(epochs_x, all_agent_arr[:, i], alpha=0.4, linewidth=0.8, color=cmap(i))
    ax.semilogy(epochs_x, best_curve, 'k-', linewidth=2, label='Mejor global')
    ax.set_title('Trayectorias de los 8 agentes')
    ax.set_xlabel('Epoch')
    ax.set_ylabel('MSE Loss (log)')
    ax.legend(fontsize=8)
    ax.grid(True, alpha=0.3)

    # Panel 2: Diversidad de agentes en el tiempo
    ax = axes[0, 1]
    if log['diversity']:
        ax.plot(epochs_x, log['diversity'], color='#3a7ebf', linewidth=1.5)
        ax.set_title('Diversidad de los agentes (varianza de pesos)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Varianza media de pesos')
        ax.grid(True, alpha=0.3)

    # Panel 3: Eventos de tunelamiento
    ax = axes[1, 0]
    if log['tunnels']:
        tunnel_arr = np.array(log['tunnels'])
        ax.bar(epochs_x, tunnel_arr, color='#e07b39', alpha=0.7, width=1.0)
        ax.set_title(f'Eventos de tunelamiento cuantico (total: {sum(log["tunnels"])})')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Agentes que tunearon')
        ax.grid(True, alpha=0.3, axis='y')

    # Panel 4: Evolucion del estado cuantico (alpha promedio)
    ax = axes[1, 1]
    if log['alphas']:
        alphas_arr = np.array(log['alphas'])  # (epochs, 8)
        ax.plot(epochs_x, alphas_arr.mean(axis=1), 'k-', linewidth=2, label='Media')
        for i in range(alphas_arr.shape[1]):
            ax.plot(epochs_x, alphas_arr[:, i], alpha=0.3, linewidth=0.7, color=cmap(i))
        ax.axhline(math.pi / 2, color='red', linestyle='--', alpha=0.5, label='pi/2 (50% tuneling)')
        ax.set_title('Estado cuantico: angulo alpha (probabilidad de tuneling)')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('alpha [0, pi]')
        ax.set_ylim(0, math.pi)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.suptitle('Diagnosticos internos de QIMADTorch (seed=42, 8 agentes, 200 epochs)', fontsize=12)
    plt.tight_layout()
    plt.savefig(PLOTS / 'diagnosticos_internos.png', dpi=150)
    plt.close()
    print("   Plot: plots/diagnosticos_internos.png")


# ══════════════════════════════════════════════════════════════════════════════
# Seccion 5: Generar RESULTADOS.md
# ══════════════════════════════════════════════════════════════════════════════

def write_report(pytest_output, pytest_summary, stats_df, wilcoxon_df, time_df, sensitivity_df, diag_stats):
    lines = []
    def w(s=''):
        lines.append(s)

    w('# Resultados: QIMADTorch — Bateria Completa de Pruebas')
    w()
    w(f'Fecha: 2026-05-29  |  Modelo: QIMADTorch v1.0  |  torch {torch.__version__}  |  Python {sys.version.split()[0]}')
    w()
    w('---')
    w()

    # ── 1. Tests unitarios ──────────────────────────────────────────────────
    w('## 1. Tests Unitarios (pytest)')
    w()
    w(f'**Resultado: {pytest_summary}**')
    w()
    w('Los tests cubren 6 categorias:')
    w()
    w('| Categoria | Tests | Descripcion |')
    w('|---|---|---|')
    w('| API | 4 | step() retorna float, n agent losses, best_obj monotono, zero_grad |')
    w('| Corrección | 5 | convergencia cuadratica, mejora vs inicial, determinismo, divergencia, best params |')
    w('| Arquitecturas | 4 | agente unico, swarm grande, modelo profundo, params congelados |')
    w('| Topologias | 4 | complete, ring, grid, random |')
    w('| Robustez numerica | 5 | distintos eta, gamma alto |')
    w('| Diagnosticos | 2 | log length, evolucion cuantica |')
    w()
    w('<details>')
    w('<summary>Salida completa de pytest</summary>')
    w()
    w('```')
    for line in pytest_output.strip().splitlines():
        w(line)
    w('```')
    w('</details>')
    w()
    w('---')
    w()

    # ── 2. Benchmark ────────────────────────────────────────────────────────
    w('## 2. Benchmark Comparativo')
    w()
    w('10 seeds independientes (42-51), 100 epochs, MLP(2->64->32->1).')
    w()
    w('### Media +/- Desviacion Estandar del MSE final')
    w()

    for task in stats_df['task'].unique():
        sub = stats_df[stats_df['task'] == task].set_index('optimizer')
        w(f'**Tarea: {task}**')
        w()
        w('| Optimizador | Media +/- Std | Mediana | Mejor | Peor |')
        w('|---|---|---|---|---|')
        for opt in ['QIMADTorch', 'Adam', 'SGD']:
            if opt in sub.index:
                row = sub.loc[opt]
                best_mark = ' **' if sub['mean'].idxmin() == opt else ''
                w(f'| {opt}{best_mark} | {row["mean_std"]} | {row["median"]:.5f} | {row["best"]:.5f} | {row["worst"]:.5f} |')
        w()

    w('### Test de Wilcoxon (QIMADTorch vs baselines, bilateral, alpha=0.05)')
    w()
    w('| Tarea | vs | p-value | Ganador |')
    w('|---|---|---|---|')
    for _, row in wilcoxon_df.iterrows():
        w(f'| {row["task"]} | {row["vs"]} | {row["p_value"]:.5f} | {row["winner"]} |')
    w()

    w('### Tiempo de computo promedio por run')
    w()
    w('| Optimizador | Segundos (100 epochs) |')
    w('|---|---|')
    for _, row in time_df.iterrows():
        w(f'| {row["optimizer"]} | {row["avg_seconds_per_run"]:.2f} s |')
    w()
    w('> QIMADTorch es mas lento que Adam porque evalua N veces el closure por step.')
    w()
    w('![Convergencia comparativa](plots/convergencia_comparativa.png)')
    w()
    w('---')
    w()

    # ── 3. Sensibilidad ─────────────────────────────────────────────────────
    w('## 3. Sensibilidad de Hiperparametros')
    w()
    w('5 seeds por configuracion, 80 epochs, tarea multimodal.')
    w()

    for param in ['num_agents', 'eta', 'topology']:
        sub = sensitivity_df[sensitivity_df['param'] == param]
        w(f'### {param}')
        w()
        w(f'| {param} | Media | Std |')
        w('|---|---|---|')
        # Sort values
        vals = sub['value'].tolist()
        try:
            vals = sorted(vals, key=float)
        except ValueError:
            pass
        for val in vals:
            row = sub[sub['value'] == val].iloc[0]
            w(f'| {val} | {row["mean"]:.5f} | {row["std"]:.5f} |')
        w()

    w('![Sensibilidad de hiperparametros](plots/sensibilidad_hiperparametros.png)')
    w()
    w('---')
    w()

    # ── 4. Diagnosticos ─────────────────────────────────────────────────────
    w('## 4. Diagnosticos Internos')
    w()
    w('Corrida unica: seed=42, 8 agentes, 200 epochs, tarea multimodal.')
    w()
    w('| Metrica | Valor |')
    w('|---|---|')
    w(f'| Eventos de tunelamiento total | {diag_stats["total_tunnels"]} |')
    w(f'| Agentes que tunelan por epoch (promedio) | {diag_stats["tunnel_per_epoch"]} |')
    w(f'| Alpha promedio inicial | {diag_stats["avg_alpha_initial"]} |')
    w(f'| Alpha promedio final | {diag_stats["avg_alpha_final"]} |')
    w(f'| Diversidad final (varianza de pesos) | {diag_stats["final_diversity"]} |')
    w(f'| Mejor MSE alcanzado | {diag_stats["best_loss"]} |')
    w()
    w('![Diagnosticos internos](plots/diagnosticos_internos.png)')
    w()
    w('---')
    w()

    # ── 5. Conclusiones ─────────────────────────────────────────────────────
    w('## 5. Conclusiones')
    w()

    # Auto-generate based on results
    for task in wilcoxon_df['task'].unique():
        task_rows = wilcoxon_df[wilcoxon_df['task'] == task]
        wins = task_rows[task_rows['winner'] == 'QIMADTorch']['vs'].tolist()
        losses = task_rows[task_rows['winner'] != 'QIMADTorch']['vs'].tolist()
        if wins:
            w(f'- **{task}**: QIMADTorch supera significativamente a {", ".join(wins)} (Wilcoxon p<0.05).')
        if losses:
            w(f'- **{task}**: QIMADTorch no supera significativamente a {", ".join(losses)}.')

    w()
    w('**Observaciones de los diagnosticos:**')
    tpe = diag_stats["tunnel_per_epoch"]
    if tpe > 2.0:
        w(f'- El tunelamiento cuantico es muy frecuente ({tpe} agentes/epoch), lo que puede indicar hiperparametros demasiado exploratorios.')
    elif tpe > 0.5:
        w(f'- El tunelamiento cuantico ocurre en {tpe} agentes/epoch — frecuencia saludable para escapar minimos locales.')
    else:
        w(f'- El tunelamiento cuantico es raro ({tpe} agentes/epoch). Aumentar alpha_lr para mayor exploracion.')
    w()
    w('**Costo computacional:** QIMADTorch es ~N veces mas lento que Adam por iteracion ')
    w('(N = num_agents). Para N=8 el costo es 8x; se recomienda usar en tareas donde ')
    w('la calidad de la solucion importa mas que la velocidad de computo.')
    w()
    w('---')
    w()
    w('*Generado automaticamente por `test_y_pruebas/run_all.py`*')

    report_path = OUTDIR / 'RESULTADOS.md'
    report_path.write_text('\n'.join(lines), encoding='utf-8')
    print(f"\nReporte generado: {report_path}")


# ══════════════════════════════════════════════════════════════════════════════
# Main
# ══════════════════════════════════════════════════════════════════════════════

def main():
    t_global = time.perf_counter()

    print("=" * 65)
    print("  QIMADTorch - Bateria completa de pruebas")
    print("=" * 65)

    pytest_output, pytest_summary = run_unit_tests()
    stats_df, wilcoxon_df, time_df, _ = run_benchmark(n_seeds=10, epochs=100)
    sensitivity_df = run_sensitivity(n_seeds=5, epochs=80)
    diag_stats = run_diagnostics(epochs=200)
    write_report(pytest_output, pytest_summary, stats_df, wilcoxon_df, time_df, sensitivity_df, diag_stats)

    total = time.perf_counter() - t_global
    print(f"\nTiempo total: {total/60:.1f} minutos")
    print(f"Archivos generados en: {OUTDIR}")
    print("  - RESULTADOS.md")
    print("  - benchmark_results.csv")
    print("  - plots/convergencia_comparativa.png")
    print("  - plots/sensibilidad_hiperparametros.png")
    print("  - plots/diagnosticos_internos.png")


if __name__ == '__main__':
    main()
