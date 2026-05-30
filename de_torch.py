"""
DETorch — Differential Evolution as a PyTorch Optimizer

Gradient-free population-based optimizer. Follows the same step(closure) API
as QIMADTorch and PSOTorch for direct comparison.

Each step calls closure() num_particles times.
DE update rule (rand/1/bin):
    mutant  = x_r1 + F * (x_r2 - x_r3)         [mutation]
    trial   = crossover(x_i, mutant, CR)         [crossover]
    x_i     = trial if f(trial) <= f(x_i) else x_i  [selection]
"""

import torch
import torch.optim


class DETorch(torch.optim.Optimizer):
    """Differential Evolution as a PyTorch optimizer.

    Args:
        params: model parameters iterator.
        num_particles (int): population size (default 8, needs >= 4).
        F (float): mutation scale factor (default 0.8). Range (0, 2].
        CR (float): crossover probability (default 0.9). Range [0, 1].
        init_scale (float): initial population spread as fraction of param
            magnitude (default 1.0).
        seed (int | None): random seed.
    """

    def __init__(self, params, num_particles=8, F=0.8, CR=0.9,
                 init_scale=1.0, seed=None):
        num_particles = max(4, num_particles)
        defaults = dict(num_particles=num_particles, F=F, CR=CR,
                        init_scale=init_scale)
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
        scale = group['init_scale']
        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        for p in all_params:
            st = self.state[p]
            init_s = float(p.data.abs().mean()) * scale + 0.1
            pop = []
            for i in range(n):
                if i == 0:
                    pop.append(p.data.clone())
                else:
                    noise = (torch.rand(p.data.shape, generator=self._rng) * 2 - 1) * init_s
                    pop.append(p.data.clone() + noise)
            st['population'] = pop
            st['fitness'] = [float('inf')] * n
            st['gbest'] = p.data.clone()

        self._fitness: list = [float('inf')] * n
        self._gbest_loss = float('inf')
        self._initialized = True

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        if not self._initialized:
            self._init_state()

        group = self.param_groups[0]
        n = group['num_particles']
        F = group['F']
        CR = group['CR']
        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]

        # Build trial population
        trials = {id(p): [] for p in all_params}
        for i in range(n):
            # Pick three distinct indices different from i
            candidates = list(range(n))
            candidates.remove(i)
            r1, r2, r3 = candidates[:3]
            # Shuffle deterministically via rng
            idxs = torch.randperm(len(candidates), generator=self._rng)[:3]
            r1, r2, r3 = [candidates[j] for j in idxs.tolist()]

            for p in all_params:
                pop = self.state[p]['population']
                # Mutation
                mutant = pop[r1] + F * (pop[r2] - pop[r3])
                # Crossover (binomial)
                mask = torch.rand(p.data.shape, generator=self._rng) < CR
                # Guarantee at least one dimension from mutant
                j_rand = torch.randint(p.data.numel(), (1,), generator=self._rng).item()
                mask.view(-1)[j_rand] = True
                trial = torch.where(mask, mutant, pop[i])
                trials[id(p)].append(trial)

        # Evaluate trials
        losses = []
        for i in range(n):
            for p in all_params:
                p.data.copy_(trials[id(p)][i])
            if p.grad is not None:
                p.grad.zero_()
            with torch.enable_grad():
                loss = closure()
            l = loss.item() if torch.is_tensor(loss) else float(loss)
            losses.append(l)

            # Selection: replace parent if trial is better or equal
            if l <= self._fitness[i]:
                self._fitness[i] = l
                for p in all_params:
                    self.state[p]['population'][i].copy_(trials[id(p)][i])

            if l < self._gbest_loss:
                self._gbest_loss = l
                for p in all_params:
                    self.state[p]['gbest'].copy_(trials[id(p)][i])

        self._last_losses = list(losses)

        # Load global best into model
        for p in all_params:
            p.data.copy_(self.state[p]['gbest'])

        return self._gbest_loss

    def get_particle_losses(self) -> list:
        return list(self._last_losses)
