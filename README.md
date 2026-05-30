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
  - cma-es
  - differential-evolution
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
| PSO baseline | `PSOTorch` en `pso_torch.py` | Enjambre sin gradiente |
| DE baseline | `DETorch` en `de_torch.py` | Differential Evolution sin gradiente |
| CMA-ES baseline | `CMAESTorch` en `cmaes_torch.py` | Estrategia evolutiva diagonal |

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

## Benchmark MNIST

> MLP 784→128→64→10 · 109K parámetros · 10 epochs · batch 512

| Optimizador | Accuracy test | Tiempo |
|---|---|---|
| **Adam (lr=1e-3)** | **97.52%** | 312 s |
| SGD + momentum | 96.58% | 293 s |
| QUIMAD 8ag k_eval=4 | 90.51% | 296 s |
| QUIMAD 4ag | 89.98% | 298 s |
| DE (8p) | 36.45% | 417 s |
| PSO (8p) | 13.83% | 320 s |

**Lectura honesta:** En MNIST con mini-batches, Adam y SGD tienen ventaja porque cada
batch es una estimación insesgada del gradiente global. QUIMAD usa el loss del batch
actual para comparar agentes, lo que introduce varianza inter-batch en el `best_theta`.
En el régimen full-batch (tareas convexa y multimodal), QUIMAD gana. PSO y DE sin
gradiente son claramente inferiores en redes con 100K+ parámetros.

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

### Cooling schedule

Reduce la exploración conforme avanza el entrenamiento — útil cuando el modelo
ya convergió a una región buena y necesita refinamiento fino:

```python
optimizer = QIMADTorch(
    model.parameters(),
    num_agents=8, eta=0.01,
    cooling='cosine',       # 'cosine' | 'linear' | 'exponential' | None
    total_steps=300,        # epochs totales
    min_temp=0.05,          # temperatura mínima al final
)
```

La temperatura decae de 1.0 a `min_temp` escalando el tamaño de los saltos de
túnel cuántico y la velocidad de rotación del estado de Bloch.

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
├── quimad_torch.py          # QIMADTorch — optimizador PyTorch (con cooling schedule)
├── pso_torch.py             # PSOTorch   — PSO como optimizador PyTorch (baseline)
├── de_torch.py              # DETorch    — Differential Evolution PyTorch (baseline)
├── cmaes_torch.py           # CMAESTorch — CMA-ES diagonal PyTorch (baseline)
├── app.py                   # Space interactivo Gradio para Hugging Face
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
│   ├── train_mlp_quimad.py  # Demo: QIMADTorch vs Adam en regresion multimodal
│   └── benchmark_mnist.py   # Benchmark MNIST: todos los optimizadores comparados
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

## Demo y Space interactivo

```bash
# QIMADTorch vs Adam en regresion multimodal (PyTorch)
python examples/train_mlp_quimad.py

# Benchmark MNIST — todos los optimizadores
python examples/benchmark_mnist.py

# Space Gradio local (o deploy en Hugging Face Spaces)
python app.py

# Simulacion 3D animada del enjambre
python simulation.py
python simulation.py --save   # guarda simulation.gif
```

El Space interactivo (`app.py`) permite elegir tarea, optimizadores e hiperparámetros
y ver curvas de convergencia en tiempo real. Deployar en HuggingFace Spaces con:
`gradio deploy` o subiendo el repo directamente.

---

## Roadmap

- [x] Implementacion NumPy con funciones de benchmark
- [x] Comparativa estadistica contra SGD, Adam, PSO (30 corridas, Wilcoxon)
- [x] Escalado a optimizador PyTorch (`QIMADTorch`)
- [x] Suite de 25 tests unitarios
- [x] Reduccion de costo con `k_eval` (evaluacion asincrona del enjambre)
- [x] PSO como baseline PyTorch (`PSOTorch`)
- [x] 9 graficas comparativas con PSO incluido
- [x] Cooling schedule coseno/lineal/exponencial (`cooling`, `total_steps`, `min_temp`)
- [x] Benchmark MNIST vs Adam/SGD/PSO/DE/CMA-ES (10 epochs, 109K parámetros)
- [x] CMA-ES diagonal y Differential Evolution como optimizadores PyTorch
- [x] Space interactivo en Hugging Face (`app.py` con Gradio)
- [ ] Benchmark CIFAR-10
- [ ] Variante full-batch-aware para mini-batch training
- [ ] Comparativa con CMA-ES en funciones benchmark de alta dimensión (D≥50)

---

## Licencia

MIT — libre para usar, modificar y distribuir.

---

**Autor: Leonardo Jiménez Martínez — Centro de Biomatemáticas BIOMAT**  
*Desarrollado como experimento de investigación en optimización inspirada en mecánica cuántica.*
