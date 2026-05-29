"""
PSOTorch — Particle Swarm Optimization as a PyTorch Optimizer

Population-based, gradient-free optimizer. Follows the same step(closure) API
as QIMADTorch so both can be compared on equal terms.

Each step calls closure() num_particles times (one per particle).
The model ends each step loaded with the best-known weights.

Usage:
    optimizer = PSOTorch(model.parameters(), num_particles=8)
    for epoch in range(epochs):
        def closure():
            optimizer.zero_grad()
            loss = criterion(model(X), y)
            loss.backward()   # backward is called but gradients are not used by PSO
            return loss
        loss = optimizer.step(closure)
"""

import torch
import torch.optim


class PSOTorch(torch.optim.Optimizer):
    """Particle Swarm Optimization as a PyTorch optimizer.

    Args:
        params: model parameters iterator.
        num_particles (int): swarm size (default 8).
        w (float): inertia weight — controls velocity decay (default 0.729).
        c1 (float): cognitive coefficient — pull toward particle's own best (default 1.494).
        c2 (float): social coefficient — pull toward global best (default 1.494).
        v_max (float | None): max velocity clamp as fraction of param magnitude.
            None disables clamping (default 0.5).
        seed (int | None): random seed (default None).

    Note: w=0.729, c1=c2=1.494 are the Clerc-Kennedy constriction factor values,
    which guarantee convergence under mild conditions.
    """

    def __init__(self, params, num_particles=8, w=0.729, c1=1.494, c2=1.494,
                 v_max=0.5, seed=None):
        defaults = dict(num_particles=num_particles, w=w, c1=c1, c2=c2, v_max=v_max)
        super().__init__(params, defaults)
        self._rng = torch.Generator()
        if seed is not None:
            self._rng.manual_seed(seed)
        self._initialized = False
        self._gbest_loss: float = float('inf')
        self._last_losses: list = []

    def _init_state(self) -> None:
        group = self.param_groups[0]
        n = group['num_particles']
        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        for p in all_params:
            st = self.state[p]
            # Scatter initial positions around the model's starting point.
            # Pure clones give zero initial diversity — PSO stalls because
            # pbest-pos = gbest-pos = 0, so only random velocity drives exploration.
            init_scale = float(p.data.abs().mean()) * 1.0 + 0.1
            st['positions'] = [
                p.data.clone() + (torch.rand(p.data.shape, generator=self._rng) * 2 - 1) * init_scale
                for _ in range(n)
            ]
            # Particle 0 starts exactly at the model init (keeps the "warm start")
            st['positions'][0] = p.data.clone()
            vel_scale = init_scale * 0.1
            st['velocities'] = [
                (torch.rand(p.data.shape, generator=self._rng) * 2 - 1) * vel_scale
                for _ in range(n)
            ]
            st['pbest'] = [pos.clone() for pos in st['positions']]
            st['gbest'] = p.data.clone()

        self._pbest_loss: list = [float('inf')] * n
        self._gbest_loss = float('inf')
        self._initialized = True

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        """Run one PSO step. Calls closure() num_particles times.

        Returns:
            float: global best loss seen so far (non-decreasing).
        """
        if not self._initialized:
            self._init_state()

        group = self.param_groups[0]
        n = group['num_particles']
        w = group['w']
        c1 = group['c1']
        c2 = group['c2']
        v_max_frac = group['v_max']

        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        losses = []

        # Phase 1: evaluate all particles
        for i in range(n):
            for p in all_params:
                p.data.copy_(self.state[p]['positions'][i])
            for p in all_params:
                if p.grad is not None:
                    p.grad.zero_()
            with torch.enable_grad():
                loss = closure()
            l = loss.item() if torch.is_tensor(loss) else float(loss)
            losses.append(l)

            # Update personal best
            if l < self._pbest_loss[i]:
                self._pbest_loss[i] = l
                for p in all_params:
                    self.state[p]['pbest'][i].copy_(self.state[p]['positions'][i])

            # Update global best
            if l < self._gbest_loss:
                self._gbest_loss = l
                for p in all_params:
                    self.state[p]['gbest'].copy_(self.state[p]['positions'][i])

        self._last_losses = list(losses)

        # Phase 2: update velocities and positions
        for i in range(n):
            r1 = torch.rand(1, generator=self._rng).item()
            r2 = torch.rand(1, generator=self._rng).item()

            for p in all_params:
                st = self.state[p]
                pos = st['positions'][i]
                vel = st['velocities'][i]
                pb  = st['pbest'][i]
                gb  = st['gbest']

                vel.mul_(w)
                vel.add_(pb - pos, alpha=c1 * r1)
                vel.add_(gb - pos, alpha=c2 * r2)

                # Velocity clamping prevents explosion
                if v_max_frac is not None:
                    v_max = float(pos.abs().mean()) * v_max_frac + 1e-4
                    vel.clamp_(-v_max, v_max)

                pos.add_(vel)

        # Phase 3: load global best into model
        for p in all_params:
            p.data.copy_(self.state[p]['gbest'])

        return self._gbest_loss

    def get_particle_losses(self) -> list:
        """Losses for all particles from the last step."""
        return list(self._last_losses)
