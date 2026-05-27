import numpy as np


class BenchmarkFunction:
    def __init__(self, name, bounds, **kwargs):
        self.name = name
        self.bounds = np.array(bounds)
        self.dim = None

    def __call__(self, x):
        raise NotImplementedError

    def gradient(self, x):
        return self.finite_difference_gradient(x)

    def finite_difference_gradient(self, x, epsilon=1e-6):
        grad = np.zeros_like(x, dtype=float)
        for i in range(len(x)):
            x_plus = x.copy()
            x_minus = x.copy()
            x_plus[i] += epsilon
            x_minus[i] -= epsilon
            grad[i] = (self(x_plus) - self(x_minus)) / (2 * epsilon)
        return grad


class Rastrigin(BenchmarkFunction):
    def __init__(self, bounds=None, A=10, **kwargs):
        super().__init__("Rastrigin", bounds if bounds is not None else [-5.12, 5.12])
        self.A = A

    def __call__(self, x):
        return self.A * self.dim + np.sum(x**2 - self.A * np.cos(2 * np.pi * x))

    def gradient(self, x):
        return 2 * x + 2 * np.pi * self.A * np.sin(2 * np.pi * x)


class Rosenbrock(BenchmarkFunction):
    def __init__(self, bounds=None, a=1.0, b=100.0, **kwargs):
        super().__init__("Rosenbrock", bounds if bounds is not None else [-5.0, 10.0])
        self.a = a
        self.b = b

    def __call__(self, x):
        if len(x) < 2:
            return (self.a - x[0])**2
        return np.sum(self.b * (x[1:] - x[:-1]**2)**2 + (self.a - x[:-1])**2)

    def gradient(self, x):
        if len(x) < 2:
            return np.array([-2 * (self.a - x[0])])
        grad = np.zeros_like(x, dtype=float)
        grad[:-1] += -2 * (self.a - x[:-1]) - 4 * self.b * x[:-1] * (x[1:] - x[:-1]**2)
        grad[1:] += 2 * self.b * (x[1:] - x[:-1]**2)
        return grad


class Ackley(BenchmarkFunction):
    def __init__(self, bounds=None, a=20, b=0.2, c=2 * np.pi, **kwargs):
        super().__init__("Ackley", bounds if bounds is not None else [-32.768, 32.768])
        self.a = a
        self.b = b
        self.c = c

    def __call__(self, x):
        t1 = -self.a * np.exp(-self.b * np.sqrt(np.sum(x**2) / self.dim))
        t2 = -np.exp(np.sum(np.cos(self.c * x)) / self.dim)
        return t1 + t2 + self.a + np.exp(1)


class SyntheticMultimodal(BenchmarkFunction):
    def __init__(self, bounds=None, num_peaks=5, **kwargs):
        super().__init__("SyntheticMultimodal", bounds if bounds is not None else [-10.0, 10.0])
        self.num_peaks = num_peaks
        self.peaks = []

    def _initialize_peaks(self, dim):
        rng = np.random.RandomState(42)
        self.peaks = []
        for _ in range(self.num_peaks):
            m = rng.uniform(self.bounds[0], self.bounds[1], dim)
            inv = np.linalg.inv(np.diag(rng.uniform(0.5, 2.0, dim)))
            amp = rng.uniform(1.0, 5.0)
            self.peaks.append({'mean': m, 'cov_inv': inv, 'amplitude': amp})

    def __call__(self, x):
        if not self.peaks:
            self._initialize_peaks(len(x))
        return (
            sum(-p['amplitude'] * np.exp(-0.5 * np.dot(x - p['mean'], np.dot(p['cov_inv'], x - p['mean'])))
                for p in self.peaks)
            + self.num_peaks * 5.0
        )


class HyperComplexSurface(BenchmarkFunction):
    """Combination of Rosenbrock valley + Rastrigin oscillations + high-frequency noise."""
    def __init__(self, bounds=None, scale=1.0, **kwargs):
        super().__init__("HyperComplexSurface", bounds if bounds is not None else [-10.0, 10.0])
        self.scale = scale

    def __call__(self, x):
        rosen = np.sum(100.0 * (x[1:] - x[:-1]**2)**2 + (1.0 - x[:-1])**2)
        rastrigin = 10 * len(x) + np.sum(x**2 - 10 * np.cos(4 * np.pi * x))
        noise = np.sum(np.sin(20 * x) * np.cos(30 * x))
        return (0.1 * rosen + rastrigin + 5 * noise) * self.scale


_REGISTRY = {
    'Rastrigin': Rastrigin,
    'Rosenbrock': Rosenbrock,
    'Ackley': Ackley,
    'SyntheticMultimodal': SyntheticMultimodal,
    'HyperComplexSurface': HyperComplexSurface,
}


def get_benchmark_function(name, dim, config_params=None):
    cls = _REGISTRY[name]
    params = {}
    if config_params and name in config_params:
        params = {k: v for k, v in config_params[name].items() if k != 'name'}
    bounds = params.pop('bounds', None)
    inst = cls(bounds=bounds, **params) if bounds is not None else cls(**params)
    inst.dim = dim
    return inst
