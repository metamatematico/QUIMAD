"""Main experiment runner for QIMAD vs. baseline comparison."""
import os

import numpy as np
import pandas as pd

from baselines import get_optimizer
from benchmarks import get_benchmark_function
from qimad_optimizer import QIMAD
from utils import load_config, plot_convergence, print_summary, save_results_to_csv


def _run_qimad(obj_func, dim, bounds, qimad_cfg, num_iterations, convergence_threshold, run_seed):
    topologies = qimad_cfg.get('topology', ['complete'])
    agents_list = qimad_cfg.get('num_agents', [8])
    skip_keys = {'name', 'num_agents', 'dimensions', 'topology'}
    base_params = {k: v for k, v in qimad_cfg.items() if k not in skip_keys}

    rows = []
    for topology in topologies:
        for num_agents in agents_list:
            opt = QIMAD(
                objective_function=obj_func,
                num_agents=num_agents,
                dim=dim,
                topology_type=topology,
                seed=run_seed,
                bounds=bounds,
                **base_params,
            )
            df = opt.optimize(num_iterations, convergence_threshold)
            df['optimizer'] = 'QIMAD'
            df['topology'] = topology
            df['num_agents'] = num_agents
            rows.append(df)
    return rows


def _run_baselines(obj_func, dim, bounds, optimizers_cfg, num_iterations, convergence_threshold):
    rows = []
    for name in ('SGD', 'Adam', 'PSO'):
        if name not in optimizers_cfg:
            continue
        cfg = dict(optimizers_cfg[name])
        cfg.pop('name', None)

        if name == 'PSO':
            particles_val = cfg.pop('num_particles', [8])
            cfg.pop('dimensions', None)
            num_p = particles_val[0] if isinstance(particles_val, list) else int(particles_val)
            cfg['num_particles'] = num_p
            n_agents = num_p
        else:
            n_agents = 1

        opt = get_optimizer(name, obj_func, dim, bounds, {name: cfg})
        df = opt.optimize(num_iterations, convergence_threshold)
        df['optimizer'] = name
        df['topology'] = 'N/A'
        df['num_agents'] = n_agents
        rows.append(df)
    return rows


def main(config_path='config.yaml'):
    cfg = load_config(config_path)
    exp = cfg['experiment']
    seed = exp.get('random_seed', 42)
    num_runs = exp.get('num_runs_per_experiment', 1)
    num_iters = exp.get('num_iterations', 150)
    conv_thr = exp.get('convergence_threshold', 1e-4)
    out_dir = exp.get('output_dir', 'results')
    plots_dir = exp.get('plots_dir', 'plots')

    qimad_cfg = cfg['optimizers'].get('QIMAD', {})
    dimensions_list = qimad_cfg.get('dimensions', [10])

    all_results = []

    for func_name, func_params in cfg['objective_functions'].items():
        bounds = func_params.get('bounds', [-10.0, 10.0])

        for dim in dimensions_list:
            obj_func = get_benchmark_function(func_name, dim, cfg['objective_functions'])

            for run in range(num_runs):
                run_seed = seed + run
                np.random.seed(run_seed)

                rows = _run_qimad(obj_func, dim, bounds, qimad_cfg,
                                  num_iters, conv_thr, run_seed)
                rows += _run_baselines(obj_func, dim, bounds, cfg['optimizers'],
                                       num_iters, conv_thr)

                for df in rows:
                    df['function'] = func_name
                    df['dimensions'] = dim
                    df['run'] = run

                all_results.extend(rows)
                print(f"  Done: {func_name} D={dim} run={run}")

    if not all_results:
        print("No results generated.")
        return pd.DataFrame()

    results_df = pd.concat(all_results, ignore_index=True)
    save_results_to_csv(results_df, out_dir)
    plot_convergence(results_df, os.path.join(out_dir, plots_dir))
    print_summary(results_df)
    return results_df


if __name__ == '__main__':
    main()
