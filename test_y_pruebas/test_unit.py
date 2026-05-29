"""
Tests unitarios para QIMADTorch.
Verifican corrección de API, convergencia básica, robustez y determinismo.

Correr desde la raiz del proyecto:
    pytest test_y_pruebas/test_unit.py -v
"""
import math
import sys
sys.path.insert(0, str(__import__('pathlib').Path(__file__).parent.parent))

import pytest
import torch
import torch.nn as nn

from quimad_torch import QIMADTorch


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_model(seed=0):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, 1))

def make_data(n=60, dim=4, seed=1):
    torch.manual_seed(seed)
    X = torch.randn(n, dim)
    y = torch.zeros(n, 1)
    return X, y

def make_closure(model, X, y, opt):
    crit = nn.MSELoss()
    def closure():
        opt.zero_grad()
        loss = crit(model(X), y)
        loss.backward()
        return loss
    return closure


# ── Bloque 1: API ─────────────────────────────────────────────────────────────

def test_step_returns_float():
    """step() debe retornar un float (convencion torch.optim)."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=2, seed=0)
    result = opt.step(make_closure(model, X, y, opt))
    assert isinstance(result, float), f"step() retorno {type(result)}, esperado float"


def test_agent_losses_length():
    """get_agent_losses() debe retornar exactamente num_agents valores."""
    for n in [1, 4, 8]:
        model, X, y = make_model(), *make_data()
        opt = QIMADTorch(model.parameters(), num_agents=n, seed=0)
        opt.step(make_closure(model, X, y, opt))
        losses = opt.get_agent_losses()
        assert len(losses) == n, f"Esperado {n} losses, obtenido {len(losses)}"


def test_best_obj_nondecreasing():
    """El mejor loss conocido nunca debe aumentar entre steps."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=4, seed=0)
    cl = make_closure(model, X, y, opt)
    prev = float('inf')
    for _ in range(15):
        loss = opt.step(cl)
        assert loss <= prev + 1e-9, f"best_obj aumento: {prev:.6f} -> {loss:.6f}"
        prev = loss


def test_zero_grad_clears_grads():
    """zero_grad() debe limpiar los gradientes del modelo."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=2, seed=0)
    crit = nn.MSELoss()
    crit(model(X), y).backward()
    opt.zero_grad()
    for p in model.parameters():
        if p.grad is not None:
            assert p.grad.abs().max().item() == 0.0, "zero_grad() no limpio los gradientes"


# ── Bloque 2: Corrección y convergencia ───────────────────────────────────────

def test_converges_on_quadratic():
    """En una funcion cuadratica simple el loss debe reducirse al menos 50%."""
    torch.manual_seed(0)
    model = nn.Linear(4, 1, bias=False)
    X = torch.randn(100, 4)
    w_true = torch.tensor([1.0, -1.0, 0.5, -0.5])
    y = (X @ w_true).unsqueeze(1)
    opt = QIMADTorch(model.parameters(), num_agents=4, eta=0.01, seed=0)
    cl = make_closure(model, X, y, opt)
    initial = opt.step(cl)
    for _ in range(99):
        final = opt.step(cl)
    assert final < initial * 0.5, (
        f"Convergencia insuficiente en cuadratica: {initial:.4f} -> {final:.4f}"
    )


def test_loss_decreases():
    """El loss final debe ser menor que el inicial en cualquier tarea."""
    model, X, y = make_model(), *make_data(n=100)
    opt = QIMADTorch(model.parameters(), num_agents=4, eta=0.01, seed=42)
    cl = make_closure(model, X, y, opt)
    initial = opt.step(cl)
    for _ in range(49):
        final = opt.step(cl)
    assert final < initial, f"Loss no mejoro: {initial:.4f} -> {final:.4f}"


def test_deterministic_with_seed():
    """Misma semilla debe producir exactamente los mismos resultados."""
    def run(seed):
        torch.manual_seed(seed)
        model = make_model(seed)
        X, y = make_data(seed=seed)
        opt = QIMADTorch(model.parameters(), num_agents=4, eta=0.01, seed=seed)
        return [opt.step(make_closure(model, X, y, opt)) for _ in range(5)]

    assert run(42) == run(42), "Resultados no son deterministicos con la misma semilla"


def test_agents_diverge():
    """Despues de varios steps, los agentes deben explorar posiciones distintas."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=8, eta=0.05, seed=7)
    cl = make_closure(model, X, y, opt)
    for _ in range(5):
        opt.step(cl)
    losses = opt.get_agent_losses()
    unique = len(set(round(l, 5) for l in losses))
    assert unique > 1, "Todos los agentes tienen el mismo loss — no estan explorando"


def test_model_has_best_params_after_step():
    """Al terminar step(), el modelo debe tener los mejores parametros conocidos."""
    torch.manual_seed(0)
    model = make_model()
    X, y = make_data(n=100)
    crit = nn.MSELoss()
    opt = QIMADTorch(model.parameters(), num_agents=4, seed=0)
    cl = make_closure(model, X, y, opt)
    for _ in range(10):
        best_reported = opt.step(cl)
    with torch.no_grad():
        current = crit(model(X), y).item()
    # El loss actual debe ser igual al mejor reportado (misma tarea, mismos datos)
    assert abs(current - best_reported) < 1e-5, (
        f"Modelo no cargado con best_theta: actual={current:.6f}, reportado={best_reported:.6f}"
    )


# ── Bloque 3: Arquitecturas ────────────────────────────────────────────────────

def test_single_agent():
    """num_agents=1 debe funcionar sin errores (RMSProp degenerado)."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=1, eta=0.01, seed=0)
    cl = make_closure(model, X, y, opt)
    for _ in range(10):
        loss = opt.step(cl)
    assert not math.isnan(loss), "NaN con num_agents=1"


def test_large_swarm():
    """num_agents=16 debe funcionar sin errores."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=16, eta=0.01, seed=0)
    cl = make_closure(model, X, y, opt)
    for _ in range(5):
        loss = opt.step(cl)
    assert not math.isnan(loss), "NaN con num_agents=16"


def test_multi_layer_deep_model():
    """Debe funcionar con modelos multi-capa profundos."""
    torch.manual_seed(0)
    model = nn.Sequential(
        nn.Linear(4, 32), nn.Tanh(),
        nn.Linear(32, 16), nn.Tanh(),
        nn.Linear(16, 8), nn.Tanh(),
        nn.Linear(8, 1),
    )
    X, y = make_data(n=80)
    opt = QIMADTorch(model.parameters(), num_agents=4, eta=0.01, seed=0)
    cl = make_closure(model, X, y, opt)
    initial = opt.step(cl)
    for _ in range(29):
        final = opt.step(cl)
    assert final < initial, "No mejoro en modelo profundo"


def test_frozen_params_unchanged():
    """Parametros con requires_grad=False no deben modificarse."""
    torch.manual_seed(0)
    model = nn.Sequential(nn.Linear(4, 8), nn.Tanh(), nn.Linear(8, 1))
    for p in model[0].parameters():
        p.requires_grad_(False)
    frozen_before = [p.data.clone() for p in model[0].parameters()]
    X, y = make_data()
    opt = QIMADTorch(model.parameters(), num_agents=2, seed=0)
    for _ in range(5):
        opt.step(make_closure(model, X, y, opt))
    for before, p in zip(frozen_before, model[0].parameters()):
        assert torch.allclose(before, p.data), "Parametro congelado fue modificado"


def test_param_count_unchanged():
    """El numero de parametros del modelo no debe cambiar durante el entrenamiento."""
    model, X, y = make_model(), *make_data()
    n_before = sum(p.numel() for p in model.parameters())
    opt = QIMADTorch(model.parameters(), num_agents=4, seed=0)
    for _ in range(5):
        opt.step(make_closure(model, X, y, opt))
    n_after = sum(p.numel() for p in model.parameters())
    assert n_before == n_after, "Numero de parametros cambio durante el entrenamiento"


# ── Bloque 4: Topologias ───────────────────────────────────────────────────────

@pytest.mark.parametrize("topology", ["complete", "ring", "grid", "random"])
def test_topology_runs_without_error(topology):
    """Todas las topologias deben correr sin errores ni NaN."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=4, topology=topology, seed=0)
    cl = make_closure(model, X, y, opt)
    for _ in range(10):
        loss = opt.step(cl)
    assert not math.isnan(loss), f"NaN con topologia '{topology}'"
    assert not math.isinf(loss), f"Inf con topologia '{topology}'"


# ── Bloque 5: Robustez numerica ────────────────────────────────────────────────

@pytest.mark.parametrize("eta", [1e-5, 1e-3, 0.05, 0.3])
def test_no_nan_across_learning_rates(eta):
    """El optimizador no debe producir NaN para un rango amplio de eta."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=4, eta=eta, seed=0)
    cl = make_closure(model, X, y, opt)
    for _ in range(10):
        loss = opt.step(cl)
    assert not math.isnan(loss), f"NaN con eta={eta}"


def test_no_nan_with_high_gamma():
    """Gamma alto (comunicacion fuerte) no debe generar NaN."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=4, gamma=0.5, seed=0)
    cl = make_closure(model, X, y, opt)
    for _ in range(10):
        loss = opt.step(cl)
    assert not math.isnan(loss), "NaN con gamma=0.5"


# ── Bloque 6: Diagnosticos ─────────────────────────────────────────────────────

def test_tracking_logs_correct_length():
    """El log de diagnosticos debe tener exactamente N entradas despues de N steps."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=4, seed=0)
    opt.enable_tracking()
    cl = make_closure(model, X, y, opt)
    for _ in range(7):
        opt.step(cl)
    log = opt.get_log()
    assert len(log['losses']) == 7, f"losses log: esperado 7, obtenido {len(log['losses'])}"
    assert len(log['tunnels']) == 7
    assert len(log['diversity']) == 7


def test_quantum_state_evolves():
    """Los angulos de la esfera de Bloch deben cambiar durante el entrenamiento."""
    model, X, y = make_model(), *make_data()
    opt = QIMADTorch(model.parameters(), num_agents=4, seed=0)
    opt.enable_tracking()
    cl = make_closure(model, X, y, opt)
    for _ in range(20):
        opt.step(cl)
    log = opt.get_log()
    alpha_first = log['alphas'][0]
    alpha_last = log['alphas'][-1]
    assert alpha_first != alpha_last, "El estado cuantico no evoluciono en 20 steps"
