import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

from benchmarks import Rastrigin, Rosenbrock, Ackley, HyperComplexSurface, get_benchmark_function
from baselines import SGD, Adam, PSO
from qimad_optimizer import QIMAD
from utils import create_topology


# --- Benchmarks ---

def test_rastrigin_minimum():
    f = Rastrigin()
    f.dim = 2
    assert f(np.zeros(2)) == pytest.approx(0.0)


def test_rosenbrock_minimum():
    f = Rosenbrock()
    f.dim = 2
    assert f(np.ones(2)) == pytest.approx(0.0, abs=1e-10)


def test_ackley_minimum():
    f = Ackley()
    f.dim = 2
    assert f(np.zeros(2)) == pytest.approx(0.0, abs=1e-10)


def test_gradient_shape():
    f = Rastrigin()
    f.dim = 5
    x = np.random.randn(5)
    g = f.gradient(x)
    assert g.shape == (5,)


def test_get_benchmark_function():
    f = get_benchmark_function('HyperComplexSurface', dim=3)
    assert f.dim == 3
    val = f(np.zeros(3))
    assert np.isfinite(val)


# --- Utils ---

def test_topology_complete():
    import networkx as nx
    G = create_topology(5, 'complete')
    assert len(G.nodes()) == 5
    assert nx.is_connected(G)


def test_topology_random():
    import networkx as nx
    G = create_topology(8, 'random', seed=0)
    assert nx.is_connected(G)


def test_topology_line():
    G = create_topology(4, 'line')
    assert len(G.nodes()) == 4


# --- Baselines ---

def _make_func(dim=5):
    f = Rastrigin()
    f.dim = dim
    return f


def test_sgd_runs():
    df = SGD(_make_func(), dim=5, bounds=[-5.12, 5.12], learning_rate=0.01).optimize(20)
    assert len(df) > 0
    assert 'best_global_objective' in df.columns


def test_adam_runs():
    df = Adam(_make_func(), dim=5, bounds=[-5.12, 5.12]).optimize(20)
    assert 'best_global_objective' in df.columns


def test_pso_runs():
    df = PSO(_make_func(), dim=5, bounds=[-5.12, 5.12], num_particles=6).optimize(20)
    assert 'diversity' in df.columns


# --- QIMAD ---

def test_qimad_runs():
    f = _make_func(dim=5)
    opt = QIMAD(f, num_agents=4, dim=5, bounds=[-5.12, 5.12], seed=0)
    df = opt.optimize(30)
    assert len(df) > 0
    assert df['best_global_objective'].iloc[-1] <= df['best_global_objective'].iloc[0]


def test_qimad_improves():
    f = Rosenbrock()
    f.dim = 3
    opt = QIMAD(f, num_agents=6, dim=3, bounds=[-5.0, 10.0], eta=0.05, seed=42)
    df = opt.optimize(100)
    assert df['best_global_objective'].iloc[-1] < df['best_global_objective'].iloc[0]


def test_qimad_columns():
    f = _make_func(dim=3)
    df = QIMAD(f, num_agents=4, dim=3, bounds=[-5.12, 5.12], seed=1).optimize(10)
    for col in ('iteration', 'mean_objective', 'best_global_objective', 'diversity'):
        assert col in df.columns
