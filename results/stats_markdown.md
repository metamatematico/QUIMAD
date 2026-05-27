# Resultados estadísticos QUIMAD

> 30 corridas independientes, semillas 42–71, D=10

## Media ± desviación estándar

| Función | Adam | PSO | QIMAD | SGD |
|---|---|---|---|---|
| Ackley | 19.4936 ± 0.2219 | **3.8863 ± 4.9000** | 10.1548 ± 2.7234 | 19.3045 ± 0.7159 |
| HyperComplexSurface | 83194.6482 ± 30826.6351 | **119.6562 ± 267.5166** | 131.6514 ± 84.5577 | 162817.0430 ± 75732.0371 |
| Rastrigin | 85.5991 ± 18.5779 | 27.7920 ± 16.3702 | **19.5845 ± 9.4554** | 122.9784 ± 22.5521 |
| Rosenbrock | 464110.7167 ± 346315.1143 | 7466.5312 ± 19992.2303 | **234.0504 ± 281.7576** | 582032.1335 ± 248992.3136 |

## Test de Wilcoxon (QUIMAD vs cada baseline)

           function   vs  p_value significant                       winner
             Ackley Adam  0.00000        Sí ✓                       QUIMAD
             Ackley  PSO  0.00001        Sí ✓                          PSO
             Ackley  SGD  0.00000        Sí ✓                       QUIMAD
HyperComplexSurface Adam  0.00000        Sí ✓                       QUIMAD
HyperComplexSurface  PSO  0.00237        Sí ✓                          PSO
HyperComplexSurface  SGD  0.00000        Sí ✓                       QUIMAD
          Rastrigin Adam  0.00000        Sí ✓                       QUIMAD
          Rastrigin  PSO  0.02341        Sí ✓                       QUIMAD
          Rastrigin  SGD  0.00000        Sí ✓                       QUIMAD
         Rosenbrock Adam  0.00000        Sí ✓                       QUIMAD
         Rosenbrock  PSO  0.88719          No Sin diferencia significativa
         Rosenbrock  SGD  0.00000        Sí ✓                       QUIMAD
