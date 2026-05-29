---
license: mit
title: QUIMAD — Quantum-Inspired Multi-Agent Descent
tags:
  - optimization
  - quantum-inspired
  - multi-agent
  - swarm-intelligence
  - pytorch
  - neural-network-optimizer
  - rmsprop
  - pso
language:
  - en
  - es
---

# QUIMAD — Quantum-Inspired Multi-Agent Descent

**Autor: Leonardo Jiménez Martínez — Centro de Biomatemáticas BIOMAT**

> *Un enjambre de esferas de Bloch rodando por una hipersuperficie rugosa.*

**QUIMAD** es un optimizador híbrido original que combina tres fuerzas simultáneas sobre cada agente del enjambre:

- **Descenso de gradiente adaptivo** (RMSProp) — cada agente baja la pendiente con paso adaptivo
- **Comunicación ponderada por fidelidad cuántica** — los agentes se escuchan en proporción a la similitud de sus estados cuánticos internos
- **Túnel cuántico** — escape estocástico de mínimos locales, con probabilidad gobernada por el estado de Bloch de cada agente

Disponible en dos implementaciones que comparten el mismo algoritmo:

| Implementación | Clase | Uso |
|---|---|---|
| PyTorch optimizer | `QIMADTorch` en `quimad_torch.py` | Entrenamiento de redes neuronales |
| NumPy puro | `QIMAD` en `qimad_optimizer.py` | Funciones de benchmark, investigación |
| PSO baseline | `PSOTorch` en `pso_torch.py` | Comparación directa sin gradiente |

---

> **Nota sobre el vocabulario cuántico**
>
> Usamos *inspirado en cuántica* para describir la **metáfora de diseño**, no el mecanismo
> de ejecución. QUIMAD **no explota superposición, entrelazamiento ni medición** en ningún
> sentido operativo: es un algoritmo completamente clásico que se ejecuta en CPU.
>
> El vocabulario cuántico nombra de forma concisa los roles estructurales del algoritmo
> y conecta a QUIMAD con la familia de metaheurísticas inspiradas en cuántica
> (QiEA, QPSO, AQGA, etc.). No se necesita hardware cuántico ni Qiskit.
> Solo `pip install -r requirements.txt`.

---

## Instalación

```bash
git clone https://github.com/metamatematico/QUIMAD.git
cd QUIMAD
pip install -r requirements.txt
```

---

## Uso rápido

### Como optimizador PyTorch (redes neuronales)

```python
import torch.nn as nn
from quimad_torch import QIMADTorch

model = nn.Sequential(nn.Linear(2, 64), nn.Tanh(), nn.Linear(64, 1))
optimizer = QIMADTorch(model.parameters(), num_agents=8, eta=0.01)

for epoch in range(300):
    def closure():
        optimizer.zero_grad()
        loss = criterion(model(X), y)
        loss.backward()
        return loss
    loss = optimizer.step(closure)
```

`step(closure)` evalúa N agentes por iteración y carga los mejores pesos al final,
siguiendo exactamente el patrón de `torch.optim.LBFGS`.

### Como optimizador NumPy (funciones de benchmark)

```python
from benchmarks import get_benchmark_function
from qimad_optimizer import QIMAD

f   = get_benchmark_function('Rastrigin', dim=10)
opt = QIMAD(f, num_agents=8, dim=10, bounds=[-5.12, 5.12])
opt.optimize(num_iterations=150)
print(f"Mejor valor encontrado: {opt.best_global_objective:.4f}")
```

---

## El algoritmo: tres fuerzas sobre cada agente

Cada agente mantiene simultáneamente:
- **θ ∈ ℝᴰ** — posición en el espacio de parámetros
- **α ∈ [0, π]** — ángulo polar de la esfera de Bloch (gobierna probabilidad de túnel)
- **β ∈ [0, 2π)** — fase de la esfera de Bloch (gobierna afinidad con otros agentes)
- **v** — acumulador RMSProp por parámetro

Actualización de posición en cada step:

```
Δθᵢ = − (η / √(v+ε)) · ∇f(θᵢ)            ← descenso adaptivo (RMSProp)
     + γ · Σⱼ F(ψᵢ,ψⱼ)^k · (θⱼ − θᵢ)     ← comunicación ponderada por fidelidad
     + salto_aleatorio  si estancado        ← túnel cuántico
```

donde `F(ψᵢ,ψⱼ) = |⟨ψᵢ|ψⱼ⟩|` es la fidelidad entre los estados de Bloch de los agentes.
Los agentes cuánticamente similares se comunican más; los divergentes exploran de forma independiente.

### ¿Por qué no es solo RMSProp ni solo PSO?

| Mecanismo | Adam / RMSProp | PSO clásico | QUIMAD |
|---|---|---|---|
| Gradiente adaptivo por agente | ✓ | ✗ | ✓ |
| Enjambre de partículas | ✗ | ✓ | ✓ |
| Peso de comunicación por fidelidad cuántica | ✗ | ✗ | **✓** |
| Probabilidad de escape dinámica por agente | ✗ | ✗ | **✓** |
| Estado cuántico acoplado a comunicación y túnel | ✗ | ✗ | **✓** |

Con `num_agents=1`, QUIMAD degenera a RMSProp puro: sin vecinos, sin comunicación,
sin túnel. La originalidad del algoritmo emerge solo con más de un agente.

---

## El estado cuántico como temperamento del explorador

El ángulo `α` controla la **probabilidad de tunelamiento**:

```
α = 0    →  P(salto) = 0    →  conservador, confía en el gradiente
α = π/2  →  P(salto) = 0.5  →  en equilibrio exploración / explotación
α = π    →  P(salto) = 1    →  temerario, siempre salta a posición aleatoria
```

El ángulo `β` determina **con quién se comunica** cada agente: la fidelidad
`F = |⟨ψᵢ|ψⱼ⟩|^k` es el peso exacto con que el agente i escucha al agente j.
Ambos ángulos evolucionan por paseo aleatorio, independiente del gradiente.

---

## Resultados: funciones de benchmark (NumPy)

> D=10 · 150 iteraciones · 8 agentes · topología completa  
> 30 corridas independientes · test de Wilcoxon bilateral (α=0.05)

| Función | QUIMAD | PSO | Adam | SGD |
|---|---|---|---|---|
| **Rastrigin** | **19.58 ± 9.46** ✓ | 27.79 ± 16.37 | 85.60 ± 18.58 | 122.98 ± 22.55 |
| **Ackley** | 10.15 ± 2.72 | **3.89 ± 4.90** ✓ | 19.49 ± 0.22 | 19.30 ± 0.72 |
| **HyperComplexSurface** | 131.65 ± 84.56 | **119.66 ± 267.52** ✓ | 83 194 ± 30 826 | 162 817 ± 75 732 |
| **Rosenbrock** | **234.05 ± 281.76** | 7 466 ± 19 992 | 464 110 ± 346 315 | 582 032 ± 248 992 |

QUIMAD gana a PSO en Rastrigin (mayor densidad de mínimos locales) y Rosenbrock.
PSO supera a QUIMAD en Ackley, pero con desviación estándar 3× mayor — menos consistente.
Ambos superan ampliamente a Adam y SGD en todos los casos multimodales.

---

## Resultados: entrenamiento de redes neuronales (PyTorch)

> 10 semillas · 120 epochs · test sobre tarea convexa y tarea multimodal

### Tarea multimodal (paisaje con muchos valles locales)

| Optimizador | Loss mediana | Costo / epoch |
|---|---|---|
| **QUIMAD full (8ag)** | **2.05** | 8 closure calls |
| QUIMAD 1 agente | 2.10 | 1 closure call |
| SGD | 2.25 | 1 closure call |
| Adam | 3.58 | 1 closure call |
| PSO (8 partículas) | 4.80 | 8 closure calls |

**QUIMAD full gana en multimodal y es el más consistente entre semillas.**
PSO paga el mismo costo computacional que QUIMAD pero sin aprovechar el gradiente.

### Tarea convexa (regresión lineal simple)

| Optimizador | Loss mediana |
|---|---|
| SGD | 0.00001 |
| QUIMAD full (8ag) | 0.00014 |
| QUIMAD 1 agente | 0.00025 |
| Adam | 0.032 |

QUIMAD es competitivo en tareas simples. Con 1 agente es equivalente a RMSProp puro.

### Reducción de costo: k_eval

```python
# ~1.7x más rápido, calidad ligeramente menor en multimodal
optimizer = QIMADTorch(model.parameters(), num_agents=8, k_eval=4, eta=0.01)
```

Evalúa solo `k_eval` agentes por step (los demás reusan gradientes cacheados).
El enjambre sigue teniendo 8 canicas — solo se reduce cuántos se actualizan por turno.

---

## Hiperparámetros

| Parámetro | Default | Efecto |
|---|---|---|
| `num_agents` | 8 | Tamaño del enjambre. Retornos decrecientes después de 4-8 |
| `eta` | 0.05 | Tasa de aprendizaje base (RMSProp). Rango útil: 1e-3 a 5e-2 |
| `gamma` | 0.05 | Fuerza de comunicación entre agentes |
| `k` | 2 | Selectividad del entrelazamiento: mayor k = comunicación más selectiva |
| `alpha_lr` | 0.03 | Velocidad de evolución del estado cuántico (eje polar) |
| `beta_lr` | 0.03 | Velocidad de evolución del estado cuántico (fase) |
| `topology` | `complete` | Grafo de comunicación: `complete`, `ring`, `grid`, `random` |
| `k_eval` | None | Agentes evaluados por step. None = todos |
| `seed` | None | Semilla para reproducibilidad |

---

## Estructura del proyecto

```
QUIMAD/
│
├── quimad_torch.py          # QIMADTorch — optimizador PyTorch para redes neuronales
├── pso_torch.py             # PSOTorch   — PSO como optimizador PyTorch (baseline)
├── qimad_optimizer.py       # QIMAD      — optimizador NumPy para benchmarks
├── baselines.py             # SGD, Adam, PSO (NumPy)
├── benchmarks.py            # Rastrigin, Rosenbrock, Ackley, HyperComplexSurface
├── utils.py                 # Topologias, plotting, I/O
├── run_experiments.py       # Orquestador de experimentos NumPy
├── statistical_analysis.py  # Tests de Wilcoxon, tablas LaTeX/Markdown
├── simulation.py            # Simulacion 3D animada del enjambre
├── config.yaml              # Parametros de experimento
│
├── examples/
│   └── train_mlp_quimad.py  # Demo: QIMADTorch vs Adam en regresion multimodal
│
├── test_y_pruebas/          # Suite de pruebas y graficas comparativas
│   ├── test_unit.py         # 25 tests unitarios del optimizador PyTorch
│   ├── run_all.py           # Runner completo con reporte automatico
│   ├── graficas_comparativas.py  # Genera 9 graficas comparativas con PSO
│   ├── RESULTADOS.md        # Reporte de resultados
│   └── *.png                # 9 graficas: convergencia, eficiencia, sensibilidad, diagnosticos
│
├── tests/
│   └── test_qimad.py        # 14 tests del optimizador NumPy
│
└── results/
    ├── experiment_results.csv
    ├── stats_markdown.md
    └── stats_latex.tex
```

---

## Tests

```bash
# Tests del optimizador PyTorch (25 tests)
pytest test_y_pruebas/test_unit.py -v

# Tests del optimizador NumPy (14 tests)
pytest tests/ -v

# Suite completa con reporte
python test_y_pruebas/run_all.py

# 9 graficas comparativas (convergencia, boxplot, eficiencia, topologias, etc.)
python test_y_pruebas/graficas_comparativas.py
```

---

## Demo

```bash
# QIMADTorch vs Adam en regresion multimodal (PyTorch)
python examples/train_mlp_quimad.py

# Simulacion 3D animada del enjambre
python simulation.py
python simulation.py --save   # guarda simulation.gif
```

---

## Roadmap

- [x] Implementacion NumPy con funciones de benchmark
- [x] Comparativa estadistica contra SGD, Adam, PSO (30 corridas, Wilcoxon)
- [x] Escalado a optimizador PyTorch (`QIMADTorch`)
- [x] Suite de 25 tests unitarios
- [x] Reduccion de costo con `k_eval` (evaluacion asincrona del enjambre)
- [x] PSO como baseline PyTorch (`PSOTorch`)
- [x] 9 graficas comparativas con PSO incluido
- [ ] Cooling schedule (reducir exploracion en iteraciones tardias)
- [ ] Benchmarks en MNIST / CIFAR-10
- [ ] Comparativa con CMA-ES y Differential Evolution
- [ ] Space interactivo en Hugging Face

---

## Licencia

MIT — libre para usar, modificar y distribuir.

---

**Autor: Leonardo Jiménez Martínez — Centro de Biomatemáticas BIOMAT**  
*Desarrollado como experimento de investigación en optimización inspirada en mecánica cuántica.*
