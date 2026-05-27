import numpy as np
import pandas as pd


class BaseOptimizer:
    def __init__(self, objective_function, dim, bounds, **kwargs):
        self.obj_func = objective_function
        self.dim = dim
        self.bounds = np.array(bounds)
        self.best_global_theta = None
        self.best_global_objective = np.inf

    def _gradient(self, theta):
        return self.obj_func.gradient(theta)

    def optimize(self, num_iterations, convergence_threshold=1e-5):
        raise NotImplementedError

    @staticmethod
    def _single_agent_row(iteration, obj_val, best):
        return {
            'iteration': iteration,
            'mean_objective': obj_val,
            'min_objective': obj_val,
            'max_objective': obj_val,
            'std_objective': 0.0,
            'best_global_objective': best,
            'diversity': 0.0,
        }


class SGD(BaseOptimizer):
    def __init__(self, objective_function, dim, bounds,
                 learning_rate=0.01, momentum=0.9, **kwargs):
        super().__init__(objective_function, dim, bounds)
        self.lr = float(learning_rate)
        self.momentum = float(momentum)
        self.theta = np.random.uniform(self.bounds[0], self.bounds[1], dim)
        self.velocity = np.zeros(dim)

    def optimize(self, num_iterations, convergence_threshold=1e-5):
        history = []
        for t in range(num_iterations):
            obj = float(self.obj_func(self.theta))
            grad = self._gradient(self.theta)

            if obj < self.best_global_objective:
                self.best_global_objective = obj
                self.best_global_theta = self.theta.copy()

            self.velocity = self.momentum * self.velocity - self.lr * grad
            self.theta = np.clip(self.theta + self.velocity, self.bounds[0], self.bounds[1])

            history.append(self._single_agent_row(t, obj, self.best_global_objective))
            if self.best_global_objective < convergence_threshold:
                break

        return pd.DataFrame(history)


class Adam(BaseOptimizer):
    def __init__(self, objective_function, dim, bounds,
                 learning_rate=0.01, beta1=0.9, beta2=0.999, epsilon=1e-8, **kwargs):
        super().__init__(objective_function, dim, bounds)
        self.lr = float(learning_rate)
        self.beta1 = float(beta1)
        self.beta2 = float(beta2)
        self.epsilon = float(epsilon)
        self.theta = np.random.uniform(self.bounds[0], self.bounds[1], dim)
        self.m = np.zeros(dim)
        self.v = np.zeros(dim)
        self.t = 0

    def optimize(self, num_iterations, convergence_threshold=1e-5):
        history = []
        for step in range(num_iterations):
            obj = float(self.obj_func(self.theta))
            grad = self._gradient(self.theta)

            if obj < self.best_global_objective:
                self.best_global_objective = obj
                self.best_global_theta = self.theta.copy()

            self.t += 1
            self.m = self.beta1 * self.m + (1 - self.beta1) * grad
            self.v = self.beta2 * self.v + (1 - self.beta2) * grad**2
            m_hat = self.m / (1 - self.beta1**self.t)
            v_hat = self.v / (1 - self.beta2**self.t)

            self.theta = np.clip(
                self.theta - self.lr * m_hat / (np.sqrt(v_hat) + self.epsilon),
                self.bounds[0], self.bounds[1]
            )

            history.append(self._single_agent_row(step, obj, self.best_global_objective))
            if self.best_global_objective < convergence_threshold:
                break

        return pd.DataFrame(history)


class PSO(BaseOptimizer):
    def __init__(self, objective_function, dim, bounds,
                 num_particles=8, w=0.7, c1=1.5, c2=1.5, **kwargs):
        super().__init__(objective_function, dim, bounds)
        self.n = int(num_particles)
        self.w = float(w)
        self.c1 = float(c1)
        self.c2 = float(c2)

        span = abs(self.bounds[1] - self.bounds[0])
        self.pos = np.random.uniform(self.bounds[0], self.bounds[1], (self.n, dim))
        self.vel = np.random.uniform(-span, span, (self.n, dim)) * 0.1
        self.pbest_pos = self.pos.copy()
        self.pbest_obj = np.array([float(self.obj_func(p)) for p in self.pos])

        best_idx = np.argmin(self.pbest_obj)
        self.best_global_objective = float(self.pbest_obj[best_idx])
        self.best_global_theta = self.pos[best_idx].copy()

    def optimize(self, num_iterations, convergence_threshold=1e-5):
        history = []
        span = abs(self.bounds[1] - self.bounds[0])

        for t in range(num_iterations):
            objs = np.array([float(self.obj_func(self.pos[i])) for i in range(self.n)])

            for i in range(self.n):
                if objs[i] < self.pbest_obj[i]:
                    self.pbest_obj[i] = objs[i]
                    self.pbest_pos[i] = self.pos[i].copy()
                if objs[i] < self.best_global_objective:
                    self.best_global_objective = objs[i]
                    self.best_global_theta = self.pos[i].copy()

            r1 = np.random.rand(self.n, self.dim)
            r2 = np.random.rand(self.n, self.dim)
            cognitive = self.c1 * r1 * (self.pbest_pos - self.pos)
            social = self.c2 * r2 * (self.best_global_theta - self.pos)
            self.vel = self.w * self.vel + cognitive + social
            self.vel = np.clip(self.vel, -span, span)
            self.pos = np.clip(self.pos + self.vel, self.bounds[0], self.bounds[1])

            history.append({
                'iteration': t,
                'mean_objective': float(objs.mean()),
                'min_objective': float(objs.min()),
                'max_objective': float(objs.max()),
                'std_objective': float(objs.std()),
                'best_global_objective': self.best_global_objective,
                'diversity': float(np.mean(np.var(self.pos, axis=0))),
            })

            if self.best_global_objective < convergence_threshold:
                break

        return pd.DataFrame(history)


_REGISTRY = {'SGD': SGD, 'Adam': Adam, 'PSO': PSO}


def get_optimizer(name, objective_function, dim, bounds, params=None):
    cls = _REGISTRY[name]
    cfg = dict(params.get(name, {})) if params else {}
    cfg.pop('name', None)
    return cls(objective_function, dim, bounds, **cfg)
