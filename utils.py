import os

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd
import yaml


def create_topology(num_agents, topology_type, seed=None):
    if topology_type == 'complete':
        return nx.complete_graph(num_agents)

    if topology_type == 'line':
        return nx.path_graph(num_agents)

    if topology_type == 'ring':
        return nx.cycle_graph(num_agents)

    if topology_type == 'grid':
        n_rows = int(np.sqrt(num_agents))
        n_cols = -(-num_agents // n_rows)  # ceiling division
        G = nx.grid_2d_graph(n_rows, n_cols)
        G = nx.convert_node_labels_to_integers(G)
        if len(G) > num_agents:
            G = nx.Graph(G.subgraph(range(num_agents)))
        return G

    if topology_type == 'random':
        if num_agents <= 1:
            G = nx.Graph()
            G.add_node(0)
            return G
        p = max(2 * np.log(num_agents) / num_agents, 0.3)
        rng_seed = seed if seed is not None else 0
        for _ in range(10):
            G = nx.erdos_renyi_graph(num_agents, min(p, 1.0), seed=rng_seed)
            if nx.is_connected(G):
                return G
            p *= 1.5
        return nx.complete_graph(num_agents)

    raise ValueError(f"Unknown topology type: {topology_type!r}")


def plot_convergence(results_df, output_dir, filename="convergence.png"):
    plt.figure(figsize=(12, 8))
    groups = results_df.groupby(['optimizer', 'function', 'topology', 'num_agents', 'dimensions'])
    for (opt, func, topo, n_agents, dim), group in groups:
        mean_obj = group.groupby('iteration')['best_global_objective'].mean()
        std_obj = group.groupby('iteration')['best_global_objective'].std().fillna(0)
        label = f'{opt} - {func} ({topo}, N={n_agents}, D={dim})'
        plt.plot(mean_obj.index, mean_obj.values, label=label)
        plt.fill_between(mean_obj.index,
                         mean_obj.values - std_obj.values,
                         mean_obj.values + std_obj.values,
                         alpha=0.15)

    plt.xlabel('Iteration')
    plt.ylabel('Best Objective Value')
    plt.title('Optimization Convergence Comparison')
    plt.yscale('log')
    plt.legend(fontsize=7)
    plt.grid(True)
    plt.tight_layout()
    os.makedirs(output_dir, exist_ok=True)
    plt.savefig(os.path.join(output_dir, filename))
    plt.close()
    print(f"Plot saved to {os.path.join(output_dir, filename)}")


def save_results_to_csv(results_df, output_dir, filename="experiment_results.csv"):
    os.makedirs(output_dir, exist_ok=True)
    filepath = os.path.join(output_dir, filename)
    results_df.to_csv(filepath, index=False)
    print(f"Results saved to {filepath}")


def load_config(config_path='config.yaml'):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)


def print_summary(results_df):
    summary = results_df.groupby(['function', 'optimizer'])['best_global_objective'].min().unstack()
    print("\n--- Experiment Summary (best value reached) ---")
    print(summary.to_string())
    print()
    for func in summary.index:
        best_opt = summary.loc[func].idxmin()
        best_val = summary.loc[func].min()
        marker = "*** QIMAD WINS ***" if best_opt == 'QIMAD' else f"Winner: {best_opt}"
        print(f"  {func}: {marker}  ({best_val:.4e})")
