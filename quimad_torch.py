"""
QIMADTorch — Quantum-Inspired Multi-Agent Descent as a PyTorch Optimizer

Extends QUIMAD from benchmark-function optimization to neural network
training. Maintains a swarm of N weight copies (agents), evaluates each
via autograd, and applies the same QUIMAD dynamics:

  RMSProp adaptive step  +  fidelity-weighted communication  +  quantum tunneling

Follows the torch.optim.LBFGS pattern: step(closure) calls closure() N times
per optimization step (once per agent). The model always ends the step loaded
with the best-known weights.

Note: designed for CPU. GPU support requires a CUDA torch.Generator.

Usage:
    optimizer = QIMADTorch(model.parameters(), num_agents=8, eta=0.01)
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()
            return loss
        loss = optimizer.step(closure)

Cost reduction:
    Pass k_eval < num_agents to evaluate only k_eval agents per step
    (round-robin rotation). Others reuse cached gradients.
    k_eval=2 with num_agents=8 gives ~4x speedup with minimal quality loss.

Cooling schedule:
    Pass total_steps to enable exploration decay. The tunneling jump scale
    and quantum-state rotation speed both decay from 1.0 to min_temp over
    total_steps, reducing exploration as training converges.
    cooling='cosine' (default) | 'linear' | 'exponential' | None (off)
"""

import math

import torch
import torch.optim


class QIMADTorch(torch.optim.Optimizer):
    """QUIMAD as a PyTorch optimizer (population-based, N forward passes per step).

    Args:
        params: model parameters iterator, same as any torch optimizer.
        num_agents (int): swarm size (default 8).
        eta (float): base learning rate for RMSProp (default 0.05).
        gamma (float): fidelity-weighted communication strength (default 0.05).
        k (int): fidelity exponent — higher = more selective entanglement (default 2).
        alpha_lr (float): tunneling-probability rotation speed (default 0.03).
        beta_lr (float): Bloch-phase rotation speed (default 0.03).
        topology (str): 'complete' | 'ring' | 'grid' | 'random' (default 'complete').
        k_eval (int | None): agents evaluated per step. None = all agents (full QUIMAD).
            Set k_eval < num_agents for cheaper steps: only k_eval closure() calls
            per step; other agents reuse cached gradients from their last turn.
            Effective cost: k_eval / num_agents times the full cost.
        seed (int | None): random seed for reproducibility (default None).
    """

    def __init__(self, params, num_agents=8, eta=0.05, gamma=0.05, k=2,
                 alpha_lr=0.03, beta_lr=0.03, topology='complete',
                 k_eval=None, seed=None,
                 cooling='cosine', total_steps=None, min_temp=0.05):
        _k = num_agents if k_eval is None else max(1, min(k_eval, num_agents))
        defaults = dict(num_agents=num_agents, eta=eta, gamma=gamma, k=k,
                        alpha_lr=alpha_lr, beta_lr=beta_lr, topology=topology,
                        k_eval=_k, cooling=cooling,
                        total_steps=total_steps, min_temp=min_temp)
        super().__init__(params, defaults)
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        self._initialized = False
        self._last_agent_losses: list = []
        self._track = False
        self._log: dict = {'losses': [], 'diversity': [], 'tunnels': [], 'alphas': [], 'betas': [], 'temperature': []}

    # ── Initialization ────────────────────────────────────────────────────────

    def _init_state(self) -> None:
        group = self.param_groups[0]
        n = group['num_agents']
        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        for g in self.param_groups:
            for p in g['params']:
                if not p.requires_grad:
                    continue
                st = self.state[p]
                st['agents_theta'] = [p.data.clone() for _ in range(n)]
                st['agents_v'] = [torch.zeros_like(p.data) for _ in range(n)]
                st['best_theta'] = p.data.clone()

        self._step_count: int = 0
        self._best_obj: float = float('inf')

        # Per-agent personal best (not global best).
        # An agent stagnates only when IT stops improving, not when another agent
        # finds a better position. This prevents excessive tunneling on easy tasks
        # and makes the single-agent case equivalent to pure RMSProp.
        self._agent_best_obj: list = [float('inf')] * n
        self._stagnation: list = [0] * n

        # Gradient cache: lets k_eval < num_agents reuse recent gradients
        self._cached_grads: dict = {
            id(p): [torch.zeros_like(p.data) for _ in range(n)]
            for p in all_params
        }
        self._cached_losses: list = [float('inf')] * n

        self._alphas = torch.rand(n, generator=self._rng) * math.pi
        self._betas = torch.rand(n, generator=self._rng) * 2 * math.pi
        self._neighbors = self._build_neighbors(n, group['topology'])
        self._initialized = True

    def _build_neighbors(self, n: int, topology: str) -> list:
        if topology == 'complete':
            return [[j for j in range(n) if j != i] for i in range(n)]
        if topology == 'ring':
            # Deduplicate and remove self-loops (edge case: n=1 gives [0,0])
            return [[nb for nb in {(i-1) % n, (i+1) % n} if nb != i] for i in range(n)]
        if topology == 'grid':
            side = math.ceil(math.sqrt(n))
            nb = []
            for i in range(n):
                r, c = divmod(i, side)
                row = []
                for dr, dc in ((-1, 0), (1, 0), (0, -1), (0, 1)):
                    j = (r + dr) * side + (c + dc)
                    if 0 <= r + dr < side and 0 <= c + dc < side and j < n and j != i:
                        row.append(j)
                nb.append(row)
            return nb
        # random: Erdős–Rényi p=0.5, guaranteed ≥1 neighbor per node
        rng = torch.Generator()
        nb = [[] for _ in range(n)]
        for i in range(n):
            for j in range(i + 1, n):
                if torch.rand(1, generator=rng).item() < 0.5:
                    nb[i].append(j)
                    nb[j].append(i)
        for i in range(n):
            if not nb[i]:
                j = (i + 1) % n
                if j != i:
                    nb[i].append(j)
                    nb[j].append(i)
        return nb

    # ── Cooling schedule ─────────────────────────────────────────────────────

    def _temperature(self) -> float:
        """Return current temperature in [min_temp, 1.0].

        Called once per step. Returns 1.0 if cooling is disabled (total_steps
        is None). Scales both the tunneling jump magnitude and the quantum-state
        rotation speed, reducing exploration as training progresses.
        """
        group = self.param_groups[0]
        total = group['total_steps']
        if total is None or total <= 0:
            return 1.0
        min_t   = group['min_temp']
        cooling = group['cooling']
        t = min(self._step_count / total, 1.0)   # progress in [0, 1]
        if cooling == 'cosine':
            return min_t + (1.0 - min_t) * 0.5 * (1.0 + math.cos(math.pi * t))
        if cooling == 'linear':
            return min_t + (1.0 - min_t) * (1.0 - t)
        if cooling == 'exponential':
            return max(min_t, math.exp(-5.0 * t))
        return 1.0   # cooling=None or unknown string → no decay

    # ── Fidelity ──────────────────────────────────────────────────────────────

    @staticmethod
    def _fidelity(ai: float, bi: float, aj: float, bj: float, k: int) -> float:
        """Compute |<psi_i|psi_j>|^k for Bloch states parametrized by (alpha, beta)."""
        re = math.cos(ai / 2) * math.cos(aj / 2) + math.cos(bj - bi) * math.sin(ai / 2) * math.sin(aj / 2)
        im = math.sin(bj - bi) * math.sin(ai / 2) * math.sin(aj / 2)
        return (re ** 2 + im ** 2) ** (k / 2)

    # ── Main step ─────────────────────────────────────────────────────────────

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        """Run one optimizer step.

        Calls closure() k_eval times per step (round-robin across agents).
        If k_eval == num_agents (default), all agents are evaluated each step.

        Returns:
            float: global best loss seen so far (non-decreasing across steps).
        """
        if not self._initialized:
            self._init_state()

        self._step_count += 1
        _n_tunnels = 0

        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        group = self.param_groups[0]
        n = group['num_agents']
        eta = group['eta']
        gamma = group['gamma']
        k = group['k']
        k_eval = group['k_eval']
        eps = 1e-8
        rms_decay = 0.99

        # Cooling schedule: temperature decays exploration, not exploitation.
        # Scales quantum-state rotation speed and tunneling jump magnitude.
        temp = self._temperature()
        alpha_lr = group['alpha_lr'] * temp
        beta_lr  = group['beta_lr']  * temp

        # ── Phase 1: Evaluate k_eval agents (round-robin), cache the rest ────
        # Agents are rotated so every agent gets a fresh evaluation every
        # ceil(num_agents / k_eval) steps.
        base = (self._step_count - 1) * k_eval % n
        to_eval = {(base + j) % n for j in range(k_eval)}

        agent_losses = list(self._cached_losses)   # start with cached values

        for i in range(n):
            if i not in to_eval:
                continue
            for p in all_params:
                p.data.copy_(self.state[p]['agents_theta'][i])
            for p in all_params:
                if p.grad is not None:
                    p.grad.zero_()
            with torch.enable_grad():
                loss = closure()
            l = loss.item() if torch.is_tensor(loss) else float(loss)
            agent_losses[i] = l
            self._cached_losses[i] = l
            for p in all_params:
                self._cached_grads[id(p)][i].copy_(
                    p.grad if p.grad is not None else torch.zeros_like(p.data)
                )

        self._last_agent_losses = list(agent_losses)
        if self._track:
            self._log['losses'].append(list(agent_losses))

        # ── Phase 2: Update global best; stagnation per agent's personal best ─
        # Stagnation counts how many steps since agent i improved *its own* best.
        # This prevents the 7 non-best agents from stagnating every single step
        # and triggering destructive tunneling on smooth loss surfaces.
        best_loss = min(agent_losses)
        best_i = agent_losses.index(best_loss)

        if best_loss < self._best_obj:
            self._best_obj = best_loss
            for p in all_params:
                self.state[p]['best_theta'].copy_(self.state[p]['agents_theta'][best_i])

        for i in range(n):
            if agent_losses[i] < self._agent_best_obj[i] - eps:
                self._agent_best_obj[i] = agent_losses[i]
                self._stagnation[i] = 0
            else:
                self._stagnation[i] += 1

        # ── Phase 3: Update each agent ────────────────────────────────────────
        alphas = self._alphas.tolist()
        betas = self._betas.tolist()

        for i in range(n):
            ai, bi = alphas[i], betas[i]
            nb = self._neighbors[i]

            # Fidelity weights for neighbors (used even for frozen agents in comm)
            if nb:
                raw_w = [self._fidelity(ai, bi, alphas[j], betas[j], k) for j in nb]
                w_sum = sum(raw_w) + eps
                norm_w = [w / w_sum for w in raw_w]
            else:
                norm_w = []

            # Random walk on Bloch sphere (all agents, every step)
            new_alpha = ai + alpha_lr * torch.randn(1, generator=self._rng).item()
            self._alphas[i] = max(0.0, min(math.pi, new_alpha))
            self._betas[i] = (bi + beta_lr * torch.randn(1, generator=self._rng).item()) % (2 * math.pi)

            # Only update position for freshly evaluated agents.
            # Agents not in to_eval keep their current position (act as fixed
            # reference points for communication until their next evaluation turn).
            if i not in to_eval:
                continue

            # Tunneling: only when swarm has > 1 agent and agent has neighbors.
            # With num_agents=1 there are no neighbors → no tunneling → pure RMSProp.
            tunnel_deltas: dict = {}
            if nb and self._stagnation[i] > 3:
                prob_jump = math.sin(ai / 2) ** 2
                if torch.rand(1, generator=self._rng).item() < prob_jump:
                    for p in all_params:
                        theta_i = self.state[p]['agents_theta'][i]
                        jump_scale = (float(theta_i.abs().mean()) * 0.8 + 0.1) * temp
                        tunnel_deltas[id(p)] = (torch.rand(theta_i.shape) * 2 - 1) * jump_scale
                    _n_tunnels += 1
                self._stagnation[i] = 0

            rand_s = torch.rand(1, generator=self._rng).item()

            for p in all_params:
                st = self.state[p]
                theta_i = st['agents_theta'][i]
                v_i = st['agents_v'][i]
                g = self._cached_grads[id(p)][i]

                # RMSProp adaptive step
                v_i.mul_(rms_decay).addcmul_(g, g, value=1 - rms_decay)
                eta_adapt = eta / (v_i.sqrt() + eps)

                # Accumulate delta from all forces before applying
                delta = -eta_adapt * g

                if nb:
                    # Attract toward global best only in multi-agent context.
                    # With a single agent best_theta == own trajectory, so this
                    # term would just add oscillation — disable it to stay pure RMSProp.
                    delta.add_(st['best_theta'] - theta_i, alpha=0.2 * rand_s)
                    comm = torch.zeros_like(theta_i)
                    for w, j in zip(norm_w, nb):
                        comm.add_(st['agents_theta'][j] - theta_i, alpha=w)
                    delta.add_(comm, alpha=gamma)

                if id(p) in tunnel_deltas:
                    delta.add_(tunnel_deltas[id(p)])

                theta_i.add_(delta)

        # ── Phase 4: Load best-known weights into the model ───────────────────
        for p in all_params:
            p.data.copy_(self.state[p]['best_theta'])

        if self._track and self._initialized:
            self._log['tunnels'].append(_n_tunnels)
            self._log['alphas'].append(self._alphas.tolist())
            self._log['betas'].append(self._betas.tolist())
            self._log['temperature'].append(temp)
            if all_params:
                thetas = torch.stack(self.state[all_params[0]]['agents_theta'])
                self._log['diversity'].append(float(thetas.var(dim=0).mean()))

        return self._best_obj

    def get_agent_losses(self) -> list:
        """Losses for all agents from the last step (diagnostics/plotting)."""
        return list(self._last_agent_losses)

    def enable_tracking(self) -> None:
        """Enable per-step diagnostic logging (slight overhead)."""
        self._track = True

    def reset_log(self) -> None:
        """Clear accumulated diagnostic log."""
        self._log = {'losses': [], 'diversity': [], 'tunnels': [], 'alphas': [], 'betas': [], 'temperature': []}

    def get_log(self) -> dict:
        """Return a copy of the diagnostic log accumulated since last reset_log()."""
        return {k: list(v) for k, v in self._log.items()}
