"""
Análisis estadístico de los resultados de QUIMAD.

Carga experiment_results.csv y produce:
  1. Tabla  media ± desviación estándar por función y optimizador
  2. Test de Wilcoxon (QUIMAD vs cada baseline, por función)
  3. Rankings con mediana y mejor valor
  4. Tabla exportable a Markdown y LaTeX

Uso:
    python statistical_analysis.py
    python statistical_analysis.py --csv results/experiment_results.csv
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


# ── Carga ────────────────────────────────────────────────────────────────────

def load(csv_path: str) -> pd.DataFrame:
    df = pd.read_csv(csv_path)
    # Para cada run, tomamos el mejor valor alcanzado en esa corrida
    best_per_run = (
        df.groupby(['function', 'optimizer', 'run'])['best_global_objective']
        .min()
        .reset_index()
        .rename(columns={'best_global_objective': 'best'})
    )
    return best_per_run


# ── Estadísticas descriptivas ────────────────────────────────────────────────

def descriptive_table(df: pd.DataFrame) -> pd.DataFrame:
    """Media, desviación estándar, mediana y mejor valor por función/optimizador."""
    stats_df = (
        df.groupby(['function', 'optimizer'])['best']
        .agg(
            runs='count',
            mean='mean',
            std='std',
            median='median',
            best='min',
            worst='max',
        )
        .round(4)
        .reset_index()
    )
    # Columna combinada mean ± std para tablas
    stats_df['mean±std'] = (
        stats_df['mean'].map('{:.4f}'.format)
        + ' ± '
        + stats_df['std'].map('{:.4f}'.format)
    )
    return stats_df


# ── Test de Wilcoxon ─────────────────────────────────────────────────────────

ALPHA = 0.05   # nivel de significancia

def wilcoxon_vs_quimad(df: pd.DataFrame) -> pd.DataFrame:
    """
    Para cada (función, baseline), aplica el test de Wilcoxon de rangos con signo
    entre QUIMAD y el baseline.

    H₀: las medianas son iguales.
    H₁: QUIMAD tiene una mediana distinta (test bilateral).

    Reporta también la dirección: QUIMAD < baseline → QUIMAD gana.
    """
    results = []
    for func in df['function'].unique():
        q_vals = df[(df['function'] == func) & (df['optimizer'] == 'QIMAD')]['best'].values
        for opt in df['optimizer'].unique():
            if opt == 'QIMAD':
                continue
            b_vals = df[(df['function'] == func) & (df['optimizer'] == opt)]['best'].values

            # Alinear longitudes (por si difieren)
            n = min(len(q_vals), len(b_vals))
            if n < 5:
                results.append(dict(function=func, vs=opt, p_value=np.nan,
                                    significant='—', winner='N/A (pocas muestras)'))
                continue

            try:
                stat, p = stats.wilcoxon(q_vals[:n], b_vals[:n], alternative='two-sided')
            except ValueError:
                # Diferencias todas cero (caso degenerado)
                results.append(dict(function=func, vs=opt, p_value=1.0,
                                    significant='No', winner='Empate'))
                continue

            significant = 'Sí ✓' if p < ALPHA else 'No'
            if p < ALPHA:
                winner = 'QUIMAD' if np.median(q_vals) < np.median(b_vals) else opt
            else:
                winner = 'Sin diferencia significativa'

            results.append(dict(
                function=func,
                vs=opt,
                p_value=round(p, 5),
                significant=significant,
                winner=winner,
            ))

    return pd.DataFrame(results)


# ── Ranking global ────────────────────────────────────────────────────────────

def ranking_table(df: pd.DataFrame) -> pd.DataFrame:
    """Número de funciones en que cada optimizador es significativamente mejor."""
    median_df = (
        df.groupby(['function', 'optimizer'])['best']
        .median()
        .reset_index()
    )
    ranks = []
    for func in median_df['function'].unique():
        sub = median_df[median_df['function'] == func].sort_values('best')
        for rank, (_, row) in enumerate(sub.iterrows(), 1):
            ranks.append({'function': func, 'optimizer': row['optimizer'], 'rank': rank})
    rank_df = pd.DataFrame(ranks)
    summary = (
        rank_df.groupby('optimizer')['rank']
        .agg(rank1=lambda x: (x == 1).sum(),
             mean_rank='mean')
        .reset_index()
        .sort_values('rank1', ascending=False)
    )
    return summary


# ── Exportar Markdown ────────────────────────────────────────────────────────

def to_markdown_table(desc: pd.DataFrame) -> str:
    """Tabla media ± std en formato Markdown, pivoteada por optimizador."""
    pivot = desc.pivot(index='function', columns='optimizer', values='mean±std')
    # Marcar el mejor (menor media) en cada fila
    medias = desc.pivot(index='function', columns='optimizer', values='mean')

    lines = []
    cols = pivot.columns.tolist()
    header = '| Función | ' + ' | '.join(cols) + ' |'
    sep    = '|---|' + '---|' * len(cols)
    lines.append(header)
    lines.append(sep)

    for func in pivot.index:
        best_opt = medias.loc[func].idxmin()
        row_parts = []
        for col in cols:
            val = pivot.loc[func, col]
            row_parts.append(f'**{val}**' if col == best_opt else val)
        lines.append('| ' + func + ' | ' + ' | '.join(row_parts) + ' |')

    return '\n'.join(lines)


def to_latex_table(desc: pd.DataFrame) -> str:
    """Tabla en formato LaTeX lista para paper."""
    pivot = desc.pivot(index='function', columns='optimizer', values='mean±std')
    medias = desc.pivot(index='function', columns='optimizer', values='mean')
    cols = pivot.columns.tolist()

    lines = [
        r'\begin{table}[h]',
        r'\centering',
        r'\caption{QUIMAD vs baselines: media $\pm$ desviación estándar (30 corridas, D=10)}',
        r'\begin{tabular}{l' + 'c' * len(cols) + '}',
        r'\hline',
        'Función & ' + ' & '.join(cols) + r' \\',
        r'\hline',
    ]
    for func in pivot.index:
        best_opt = medias.loc[func].idxmin()
        parts = []
        for col in cols:
            val = pivot.loc[func, col].replace('±', r'$\pm$')
            parts.append(r'\textbf{' + val + '}' if col == best_opt else val)
        lines.append(func + ' & ' + ' & '.join(parts) + r' \\')
    lines += [r'\hline', r'\end{tabular}', r'\end{table}']
    return '\n'.join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main(csv_path='results/experiment_results.csv'):
    print(f'\nCargando: {csv_path}')
    df = load(csv_path)

    n_runs = df['run'].nunique()
    opts   = df['optimizer'].unique().tolist()
    funcs  = df['function'].unique().tolist()
    print(f'Corridas por experimento : {n_runs}')
    print(f'Optimizadores            : {opts}')
    print(f'Funciones                : {funcs}\n')

    # ── 1. Estadísticas descriptivas ─────────────────────────────────────────
    desc = descriptive_table(df)
    print('=' * 70)
    print('TABLA 1 — Media ± Desviación estándar (mejor valor por corrida)')
    print('=' * 70)
    pivot_display = desc.pivot(index='function', columns='optimizer',
                               values='mean±std')
    print(pivot_display.to_string())

    # ── 2. Test de Wilcoxon ──────────────────────────────────────────────────
    wdf = wilcoxon_vs_quimad(df)
    print('\n' + '=' * 70)
    print(f'TABLA 2 — Test de Wilcoxon: QUIMAD vs baselines  (alpha = {ALPHA})')
    print('=' * 70)
    print(wdf.to_string(index=False))

    # ── 3. Ranking ───────────────────────────────────────────────────────────
    rank = ranking_table(df)
    print('\n' + '=' * 70)
    print('TABLA 3 — Ranking global (veces en 1er lugar por mediana)')
    print('=' * 70)
    print(rank.to_string(index=False))

    # ── 4. Exportar ──────────────────────────────────────────────────────────
    md_table = to_markdown_table(desc)
    latex_table = to_latex_table(desc)

    Path('results').mkdir(exist_ok=True)
    with open('results/stats_markdown.md', 'w', encoding='utf-8') as f:
        f.write('# Resultados estadísticos QUIMAD\n\n')
        f.write(f'> {n_runs} corridas independientes, semillas {42}–{42+n_runs-1}, D=10\n\n')
        f.write('## Media ± desviación estándar\n\n')
        f.write(md_table + '\n\n')
        f.write('## Test de Wilcoxon (QUIMAD vs cada baseline)\n\n')
        # to_markdown requiere tabulate; usamos to_string como fallback
        try:
            f.write(wdf.to_markdown(index=False) + '\n')
        except ImportError:
            f.write(wdf.to_string(index=False) + '\n')

    with open('results/stats_latex.tex', 'w', encoding='utf-8') as f:
        f.write(latex_table)

    print('\nArchivos generados:')
    print('  results/stats_markdown.md')
    print('  results/stats_latex.tex')

    return desc, wdf, rank


if __name__ == '__main__':
    ap = argparse.ArgumentParser()
    ap.add_argument('--csv', default='results/experiment_results.csv')
    args = ap.parse_args()
    main(args.csv)
