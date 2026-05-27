"""
QIMAD — Quantum-Inspired Multi-Agent Descent
=============================================

Origen conceptual
-----------------
QIMAD nace de la siguiente imagen física:

    Un conjunto de canicas cuánticas —esferas de Bloch— rueda
    por una hipersuperficie no trivial. Cada canica tiene dos
    capas de descripción simultáneas:

      - Capa clásica : posición θ ∈ ℝᴰ sobre la hipersuperficie,
                       que evoluciona siguiendo la pendiente (gradiente).

      - Capa cuántica: orientación interna como esfera de Bloch,
                       parametrizada por (α, β), que cambia conforme
                       la canica rueda — la curvatura local del terreno
                       genera rotación del estado interno.

El mecanismo clave es el umbral de entrelazamiento:

    Cuando la fidelidad F(ψᵢ, ψⱼ) = |⟨ψᵢ|ψⱼ⟩|² supera un umbral,
    dos canicas se entrelazan y abren un canal de información.
    Bajo el umbral son independientes; sobre él comparten posición,
    gradiente local y calidad del mínimo encontrado.

    Esta auto-organización emergente hace que:
      - Canicas en regiones distintas (estados cuánticos divergentes)
        exploren independientemente sin interferirse.
      - Canicas convergiendo al mismo mínimo (estados cuánticos
        alineados) colaboren para refinarlo.

Aproximaciones en esta implementación
--------------------------------------
- La rotación inducida por la hipersuperficie se aproxima como
  caminata aleatoria acotada sobre (α, β) con tasas alpha_lr / beta_lr.
- El entrelazamiento binario se aproxima como ponderación continua
  por fidelidad F^k: a mayor k, más selectiva (más parecida a un
  umbral duro).
- El canal de información se realiza mediante el término de
  comunicación ponderado: γ · Σ w_ij · (θⱼ − θᵢ).
"""

import networkx as nx
import numpy as np
import pandas as pd

from utils import create_topology


class QIMADAgent:
    def __init__(self, dim, bounds, rng=None):
        rng = rng or np.random
        self.dim = dim
        self.bounds = bounds
        self.theta = rng.uniform(bounds[0], bounds[1], dim)
        self.alpha = rng.uniform(0, np.pi)
        self.beta = rng.uniform(0, 2 * np.pi)
        self.psi = self._compute_psi()
        self.v = np.zeros(dim)          # RMSProp accumulator
        self.stagnation_counter = 0

    def _compute_psi(self):
        return np.array([
            np.cos(self.alpha / 2),
            np.exp(1j * self.beta) * np.sin(self.alpha / 2)
        ], dtype=complex)

    def update_quantum_state(self, alpha_lr, beta_lr, rng):
        self.alpha = np.clip(self.alpha + alpha_lr * rng.randn(), 0, np.pi)
        self.beta = (self.beta + beta_lr * rng.randn()) % (2 * np.pi)
        self.psi = self._compute_psi()


class QIMAD:
    """Quantum-Inspired Multi-Agent Descent optimizer."""

    def __init__(self, objective_function, num_agents=8, dim=10,
                 eta=0.05, gamma=0.05, k=2,
                 alpha_lr=0.03, beta_lr=0.03,
                 entanglement_strength=0.05,
                 topology_type='complete',
                 seed=None, bounds=None, **kwargs):
        self.obj_func = objective_function
        self.num_agents = num_agents
        self.dim = dim
        self.eta = eta
        self.gamma = gamma
        self.k = k
        self.alpha_lr = alpha_lr
        self.beta_lr = beta_lr
        self.entanglement_strength = entanglement_strength
        self.bounds = np.array(bounds) if bounds is not None else np.array([-10.0, 10.0])
        self.eps = 1e-8

        rng = np.random.RandomState(seed)
        self.rng = rng
        self.agents = [QIMADAgent(dim, self.bounds, rng) for _ in range(num_agents)]

        if topology_type == 'complete':
            self.G = nx.complete_graph(num_agents)
        else:
            self.G = create_topology(num_agents, topology_type, seed)

        self.best_global_theta = None
        self.best_global_objective = np.inf

    def _fidelity_weight(self, psi_i, psi_j):
        return np.abs(np.vdot(psi_i, psi_j)) ** self.k

    def optimize(self, num_iterations, convergence_threshold=1e-5):
        history = []
        rmsprop_decay = 0.99

        for iteration in range(num_iterations):
            current_objs = [self.obj_func(a.theta) for a in self.agents]

            # Update global best and stagnation counters
            for i, obj in enumerate(current_objs):
                if obj < self.best_global_objective:
                    self.best_global_objective = obj
                    self.best_global_theta = self.agents[i].theta.copy()
                    self.agents[i].stagnation_counter = 0
                else:
                    self.agents[i].stagnation_counter += 1

            # Update each agent
            for i, agent in enumerate(self.agents):
                grad = self.obj_func.gradient(agent.theta)

                # RMSProp adaptive learning rate
                agent.v = rmsprop_decay * agent.v + (1 - rmsprop_decay) * grad**2
                adaptive_eta = self.eta / (np.sqrt(agent.v) + self.eps)

                # Global attraction toward best known position
                attraction = 0.2 * self.rng.rand() * (self.best_global_theta - agent.theta)

                # Quantum tunneling: escape stagnation with a random jump
                tunneling = np.zeros(self.dim)
                if agent.stagnation_counter > 3:
                    prob_jump = float(np.abs(agent.psi[1])**2)
                    if self.rng.rand() < prob_jump:
                        jump_scale = (self.bounds[1] - self.bounds[0]) * 0.4
                        tunneling = self.rng.uniform(-1, 1, self.dim) * jump_scale
                    agent.stagnation_counter = 0

                # Fidelity-weighted communication with neighbors
                comm = np.zeros(self.dim)
                neighbors = list(self.G.neighbors(i))
                if neighbors:
                    weights = np.array([self._fidelity_weight(agent.psi, self.agents[n].psi)
                                        for n in neighbors])
                    norm_w = weights / (weights.sum() + self.eps)
                    for w, n_idx in zip(norm_w, neighbors):
                        comm += w * (self.agents[n_idx].theta - agent.theta)

                agent.theta = np.clip(
                    agent.theta - adaptive_eta * grad + self.gamma * comm + attraction + tunneling,
                    self.bounds[0], self.bounds[1]
                )
                agent.update_quantum_state(self.alpha_lr, self.beta_lr, self.rng)

            history.append({
                'iteration': iteration,
                'mean_objective': float(np.mean(current_objs)),
                'min_objective': float(np.min(current_objs)),
                'max_objective': float(np.max(current_objs)),
                'std_objective': float(np.std(current_objs)),
                'best_global_objective': float(self.best_global_objective),
                'diversity': float(np.mean(np.var([a.theta for a in self.agents], axis=0))),
            })

            if self.best_global_objective < convergence_threshold:
                break

        return pd.DataFrame(history)
