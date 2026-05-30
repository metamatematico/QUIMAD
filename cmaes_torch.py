"""
CMAESTorch — Diagonal CMA-ES as a PyTorch Optimizer

Gradient-free evolutionary strategy. Uses a diagonal covariance approximation
(sep-CMA-ES style) to keep memory O(D) instead of O(D²), making it practical
for neural networks up to ~10K parameters.

Each step calls closure() lambda_ times (population size).

Reference: Hansen (2016) "The CMA Evolution Strategy: A Tutorial"
"""

import math

import torch
import torch.optim


class CMAESTorch(torch.optim.Optimizer):
    """Diagonal CMA-ES as a PyTorch optimizer.

    Args:
        params: model parameters iterator.
        sigma0 (float): initial step size (default 0.3).
        lambda_ (int | None): population size. None → 4 + floor(3*ln(D)).
        mu_frac (float): fraction of population selected as parents (default 0.5).
        seed (int | None): random seed.
    """

    def __init__(self, params, sigma0=0.3, lambda_=None, mu_frac=0.5, seed=None):
        defaults = dict(sigma0=sigma0, lambda_=lambda_, mu_frac=mu_frac)
        super().__init__(params, defaults)
        self._rng_gen = torch.Generator()
        if seed is not None:
            self._rng_gen.manual_seed(seed)
        self._initialized = False
        self._gbest_loss: float = float('inf')
        self._last_losses: list = []

    def _flatten(self, all_params) -> torch.Tensor:
        return torch.cat([p.data.view(-1) for p in all_params])

    def _unflatten(self, vec: torch.Tensor, all_params) -> None:
        offset = 0
        for p in all_params:
            n = p.data.numel()
            p.data.copy_(vec[offset:offset + n].view(p.data.shape))
            offset += n

    def _init_state(self) -> None:
        group = self.param_groups[0]
        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        D = sum(p.data.numel() for p in all_params)

        lam = group['lambda_']
        if lam is None:
            lam = int(4 + math.floor(3 * math.log(max(D, 1))))
        lam = max(lam, 4)

        mu = max(1, int(lam * group['mu_frac']))

        # Recombination weights (log-linear)
        weights = torch.log(torch.tensor(mu + 0.5)) - torch.log(
            torch.arange(1, mu + 1, dtype=torch.float32))
        weights = weights / weights.sum()
        mu_eff = float(1.0 / (weights ** 2).sum())

        # Step-size control constants
        cs = (mu_eff + 2) / (D + mu_eff + 5)
        ds = 1 + 2 * max(0, math.sqrt((mu_eff - 1) / (D + 1)) - 1) + cs
        chi_n = math.sqrt(D) * (1 - 1 / (4 * D) + 1 / (21 * D ** 2))

        # Covariance rank-one / rank-mu constants (diagonal only)
        cc = (4 + mu_eff / D) / (D + 4 + 2 * mu_eff / D)
        c1 = 2 / ((D + 1.3) ** 2 + mu_eff)
        cmu = min(1 - c1, 2 * (mu_eff - 2 + 1 / mu_eff) / ((D + 2) ** 2 + mu_eff))

        self._D = D
        self._lam = lam
        self._mu = mu
        self._weights = weights
        self._mu_eff = mu_eff
        self._cs = cs
        self._ds = ds
        self._chi_n = chi_n
        self._cc = cc
        self._c1 = c1
        self._cmu = cmu

        self._mean = self._flatten(all_params)
        self._sigma = group['sigma0']
        self._ps = torch.zeros(D)    # evolution path step-size
        self._pc = torch.zeros(D)    # evolution path covariance
        self._C_diag = torch.ones(D) # diagonal covariance
        self._gbest = self._mean.clone()
        self._gbest_loss = float('inf')
        self._initialized = True

    @torch.no_grad()
    def step(self, closure):  # type: ignore[override]
        if not self._initialized:
            self._init_state()

        all_params = [p for g in self.param_groups for p in g['params'] if p.requires_grad]
        D = self._D
        lam = self._lam
        mu = self._mu

        # Sample population: x_k = mean + sigma * C_diag^0.5 * z_k
        C_sqrt = self._C_diag.sqrt()
        zs = []
        xs = []
        for _ in range(lam):
            z = torch.randn(D, generator=self._rng_gen)
            x = self._mean + self._sigma * C_sqrt * z
            zs.append(z)
            xs.append(x)

        # Evaluate
        losses = []
        for i in range(lam):
            self._unflatten(xs[i], all_params)
            for p in all_params:
                if p.grad is not None:
                    p.grad.zero_()
            with torch.enable_grad():
                loss = closure()
            l = loss.item() if torch.is_tensor(loss) else float(loss)
            losses.append(l)
            if l < self._gbest_loss:
                self._gbest_loss = l
                self._gbest.copy_(xs[i])

        self._last_losses = list(losses)

        # Sort by fitness (ascending)
        order = sorted(range(lam), key=lambda i: losses[i])
        best_zs = torch.stack([zs[order[j]] for j in range(mu)])  # (mu, D)
        best_xs = torch.stack([xs[order[j]] for j in range(mu)])  # (mu, D)

        # Update mean
        old_mean = self._mean.clone()
        self._mean = (self._weights.unsqueeze(1) * best_xs).sum(dim=0)

        step_vec = (self._mean - old_mean) / self._sigma   # normalized step

        # Update evolution path for step-size control
        self._ps = ((1 - self._cs) * self._ps
                    + math.sqrt(self._cs * (2 - self._cs) * self._mu_eff) * step_vec)

        # Update evolution path for covariance
        h_sig = float(self._ps.norm()) / math.sqrt(
            1 - (1 - self._cs) ** (2 * (len(self._last_losses) + 1))) < (
                1.4 + 2 / (D + 1)) * self._chi_n
        self._pc = ((1 - self._cc) * self._pc
                    + (h_sig * math.sqrt(self._cc * (2 - self._cc) * self._mu_eff)) * step_vec)

        # Update diagonal covariance
        rank_one = self._pc ** 2
        rank_mu = (self._weights.unsqueeze(1) * best_zs ** 2).sum(dim=0)
        self._C_diag = ((1 - self._c1 - self._cmu) * self._C_diag
                        + self._c1 * rank_one
                        + self._cmu * rank_mu)
        self._C_diag.clamp_(min=1e-20)

        # Update step size (Cumulative Step-size Adaptation)
        self._sigma *= math.exp((self._cs / self._ds)
                                * (float(self._ps.norm()) / self._chi_n - 1))
        self._sigma = max(1e-10, min(self._sigma, 1e3))

        # Load global best into model
        self._unflatten(self._gbest, all_params)
        return self._gbest_loss

    def get_particle_losses(self) -> list:
        return list(self._last_losses)
