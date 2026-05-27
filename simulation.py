"""
QIMAD Visual Simulation
=======================
Canicas cuánticas (esferas de Bloch) rodando sobre una hipersuperficie rugosa.

Paneles:
  - Principal 3D : la superficie f: ℝ² → ℝ con las canicas moviéndose
  - Convergencia : evolución de f_min a lo largo del tiempo
  - Bloch 2D     : proyección ecuatorial del estado cuántico de cada canica

Efectos visuales:
  - Color de la canica  → ángulo α (azul = conservador, rojo = temerario)
  - Tamaño de la canica → calidad de la posición (mejor → más grande)
  - Líneas doradas      → pares entrelazados (F > umbral)
  - Flash blanco        → evento de túnel cuántico

Uso:
    python simulation.py               # ventana interactiva
    python simulation.py --save        # guarda simulation.gif (requiere Pillow)
    python simulation.py --agents 12 --iters 150 --seed 7
"""

import argparse

import matplotlib
import matplotlib.gridspec as gridspec
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import cm
from matplotlib.animation import FuncAnimation

# ── Colores del tema ─────────────────────────────────────────────────────────

_BG       = '#0b0c10'
_PANEL    = '#0f1117'
_GRID     = '#1a1d2e'
_TEXT     = '#c8d0e0'
_CYAN     = '#00e5ff'
_ORANGE   = '#ff6b35'
_GOLD     = '#ffd060'
_DIM      = '#444466'

# ── Superficie f: ℝ² → ℝ ────────────────────────────────────────────────────

BOUNDS = [-6.0, 6.0]


def rugose(x, y):
    """
    Superficie rugosa no trivial: combinación de cuenco suave,
    crestas periódicas y ruido de alta frecuencia.

    Diseñada para tener múltiples mínimos locales y una estructura
    de valles que no es trivialmente navegable por gradiente.
    """
    bowl   = 0.75 * (x**2 + y**2)
    ridges = 2.5  * np.tan(2.0 * x) * np.cos(2.0 * y)
    fine   = 0.7  * np.sin(5.2 * x) * np.tan(5.2 * y)
    return bowl + ridges + fine


class Surface2D:
    """Wrapper que expone la superficie como función objetivo para QIMAD."""
    dim   = 4
    bounds = np.array(BOUNDS, dtype=float)

    def __call__(self, x):
        return float(rugose(x[0], x[1]))

    def gradient(self, x, eps=1e-5):
        g = np.zeros(2)
        for i in range(2):
            xp, xm = x.copy(), x.copy()
            xp[i] += eps
            xm[i] -= eps
            g[i] = (self(xp) - self(xm)) / (2 * eps)
        return g


# ── Simulador QIMAD que almacena historia completa ───────────────────────────

class QIMADSimulator:
    """
    Variante de QIMAD reducida a ℝ² que registra el estado completo
    en cada iteración para reproducción en la animación.

    Origen del modelo
    -----------------
    Cada agente es una esfera de Bloch que rueda por la hipersuperficie.
    Al rodar, su estado cuántico (α, β) evoluciona. Cuando dos esferas
    tienen estados cercanos (alta fidelidad), se entrelazan y abren un
    canal de información entre sus posiciones clásicas.
    """

    ENTANGLEMENT_THRESHOLD = 0.55   # umbral de fidelidad para entrelazamiento

    def __init__(self, obj, num_agents=8, eta=0.05, gamma=0.12,
                 k=2, alpha_lr=0.06, beta_lr=0.06, seed=42):
        import networkx as nx

        self.obj       = obj
        self.N         = num_agents
        self.eta       = eta
        self.gamma     = gamma
        self.k         = k
        self.alpha_lr  = alpha_lr
        self.beta_lr   = beta_lr
        self.rng       = np.random.RandomState(seed)
        self.bounds    = obj.bounds
        self.eps       = 1e-8
        self.G         = nx.complete_graph(num_agents)

        # Estado inicial de las canicas
        self.pos       = self.rng.uniform(*self.bounds, (num_agents, 2))
        self.alpha     = self.rng.uniform(0, np.pi, num_agents)
        self.beta      = self.rng.uniform(0, 2 * np.pi, num_agents)
        self.v_rms     = np.zeros((num_agents, 2))
        self.stagnation = np.zeros(num_agents, dtype=int)
        self.best_pos  = self.pos[0].copy()
        self.best_val  = np.inf

        # Historiales (uno por iteración)
        self.h_pos      = []   # (N, 2) posiciones clásicas
        self.h_alpha    = []   # (N,)  ángulo α de cada esfera
        self.h_beta     = []   # (N,)  ángulo β de cada esfera
        self.h_obj      = []   # (N,)  valor objetivo de cada agente
        self.h_best     = []   # float mejor global acumulado
        self.h_fidelity = []   # (N, N) matriz de fidelidades
        self.h_entangled = []  # lista de pares (i, j) entrelazados
        self.h_tunneled  = []  # índices de agentes que tunelizan en este paso

    # ── estado cuántico ──────────────────────────────────────────────────────

    def _psi(self, i):
        """Qubit en representación de la esfera de Bloch."""
        a, b = self.alpha[i], self.beta[i]
        return np.array([
            np.cos(a / 2),
            np.exp(1j * b) * np.sin(a / 2)
        ], dtype=complex)

    def _fidelity_matrix(self):
        psis = [self._psi(i) for i in range(self.N)]
        return np.array([
            [abs(np.vdot(psis[i], psis[j]))**2 for j in range(self.N)]
            for i in range(self.N)
        ])

    # ── un paso de optimización ──────────────────────────────────────────────

    def step(self):
        objs = np.array([self.obj(self.pos[i]) for i in range(self.N)])

        for i, v in enumerate(objs):
            if v < self.best_val:
                self.best_val = v
                self.best_pos = self.pos[i].copy()
                self.stagnation[i] = 0
            else:
                self.stagnation[i] += 1

        F = self._fidelity_matrix()
        entangled = [
            (i, j) for i in range(self.N)
            for j in range(i + 1, self.N)
            if F[i, j] > self.ENTANGLEMENT_THRESHOLD
        ]

        # Guardar estado ANTES de actualizar posiciones
        self.h_pos.append(self.pos.copy())
        self.h_alpha.append(self.alpha.copy())
        self.h_beta.append(self.beta.copy())
        self.h_obj.append(objs.copy())
        self.h_best.append(self.best_val)
        self.h_fidelity.append(F.copy())
        self.h_entangled.append(entangled)

        # Actualizar cada canica
        tunneled = []
        new_pos = self.pos.copy()
        for i in range(self.N):
            g = self.obj.gradient(self.pos[i])
            self.v_rms[i] = 0.99 * self.v_rms[i] + 0.01 * g**2
            eta_a = self.eta / (np.sqrt(self.v_rms[i]) + self.eps)

            # Atracción hacia el mejor conocido
            attract = 0.2 * self.rng.rand() * (self.best_pos - self.pos[i])

            # Túnel cuántico: la amplitud |1⟩ dicta la probabilidad de salto
            tunnel = np.zeros(2)
            if self.stagnation[i] > 3:
                prob = float(abs(self._psi(i)[1])**2)   # sin²(α/2)
                if self.rng.rand() < prob:
                    span = self.bounds[1] - self.bounds[0]
                    tunnel = self.rng.uniform(-1, 1, 2) * span * 0.4
                    tunneled.append(i)
                self.stagnation[i] = 0

            # Comunicación ponderada por fidelidad (entrelazamiento)
            neighbors = list(self.G.neighbors(i))
            weights = F[i, neighbors] ** self.k
            comm = np.zeros(2)
            if weights.sum() > self.eps:
                nw = weights / weights.sum()
                for w, n in zip(nw, neighbors):
                    comm += w * (self.pos[n] - self.pos[i])

            new_pos[i] = np.clip(
                self.pos[i] - eta_a * g + self.gamma * comm + attract + tunnel,
                self.bounds[0], self.bounds[1]
            )

            # La canica rueda → su estado de Bloch evoluciona
            self.alpha[i] = np.clip(
                self.alpha[i] + self.alpha_lr * self.rng.randn(), 0, np.pi)
            self.beta[i] = (
                self.beta[i] + self.beta_lr * self.rng.randn()) % (2 * np.pi)

        self.h_tunneled.append(tunneled)
        self.pos = new_pos

    def run(self, n_iters):
        for _ in range(n_iters):
            self.step()


# ── Animación ────────────────────────────────────────────────────────────────

def _build_mesh(n=100):
    x = np.linspace(BOUNDS[0], BOUNDS[1], n)
    y = np.linspace(BOUNDS[0], BOUNDS[1], n)
    X, Y = np.meshgrid(x, y)
    return X, Y, rugose(X, Y)


def run_simulation(num_agents=8, num_iters=120, seed=42, save_gif=False):
    # ── Pre-calcular toda la trayectoria ─────────────────────────────────────
    print(f"Calculando {num_iters} iteraciones con {num_agents} canicas...")
    obj = Surface2D()
    sim = QIMADSimulator(obj, num_agents=num_agents, seed=seed)
    sim.run(num_iters)
    print("Preparando animación...")

    X, Y, Z = _build_mesh()
    z_lo = Z.min() - 0.3
    z_hi = Z.max() + 0.9
    cmap_q = matplotlib.colormaps.get_cmap('plasma')

    # ── Figura ────────────────────────────────────────────────────────────────
    fig = plt.figure(figsize=(17, 9), facecolor=_BG)
    fig.suptitle(
        'QIMAD  ·  Canicas cuánticas (esferas de Bloch) rodando '
        'sobre una hipersuperficie rugosa',
        color=_TEXT, fontsize=12, fontweight='bold', y=0.985
    )

    gs = gridspec.GridSpec(
        2, 3, figure=fig,
        width_ratios=[3, 1.3, 1.1],
        hspace=0.42, wspace=0.32,
        left=0.03, right=0.97, top=0.935, bottom=0.07
    )
    ax3d  = fig.add_subplot(gs[:, 0], projection='3d')
    ax_cv = fig.add_subplot(gs[0, 1])
    ax_bl = fig.add_subplot(gs[1, 1])
    ax_lg = fig.add_subplot(gs[:, 2])

    for ax in (ax_cv, ax_bl, ax_lg):
        ax.set_facecolor(_PANEL)
        for sp in ax.spines.values():
            sp.set_color(_GRID)

    # ── Panel 3D ─────────────────────────────────────────────────────────────
    ax3d.set_facecolor(_BG)
    for axis in (ax3d.xaxis, ax3d.yaxis, ax3d.zaxis):
        axis.pane.fill = False
        axis.line.set_color('#222233')
    ax3d.tick_params(colors='#444455', labelsize=6)

    ax3d.plot_surface(X, Y, Z, cmap='viridis', alpha=0.42,
                      linewidth=0, antialiased=True, rcount=80, ccount=80)

    ax3d.set_xlim(BOUNDS); ax3d.set_ylim(BOUNDS); ax3d.set_zlim(z_lo, z_hi)
    ax3d.set_xlabel('x', color='#8888aa', labelpad=1, fontsize=7)
    ax3d.set_ylabel('y', color='#8888aa', labelpad=1, fontsize=7)
    ax3d.set_zlabel('f(x,y)', color='#8888aa', labelpad=1, fontsize=7)
    ax3d.view_init(elev=28, azim=-52)

    # Scatter de agentes (inicializado con posiciones reales del primer frame)
    pos0   = sim.h_pos[0]
    zs0    = np.array([rugose(pos0[i, 0], pos0[i, 1]) for i in range(num_agents)])
    colors0 = cmap_q(sim.h_alpha[0] / np.pi)
    scat = ax3d.scatter(
        pos0[:, 0], pos0[:, 1], zs0 + 0.08,
        s=120, c=colors0, depthshade=True, zorder=10
    )

    iter_txt = ax3d.text2D(0.02, 0.96, '', transform=ax3d.transAxes,
                           color=_TEXT, fontsize=9, va='top')
    best_txt = ax3d.text2D(0.02, 0.89, '', transform=ax3d.transAxes,
                           color=_CYAN, fontsize=8, va='top')

    # Rastros (últimos TRAIL pasos de cada canica)
    TRAIL = 8
    trail_lines = []
    for i in range(num_agents):
        ln, = ax3d.plot([], [], [], lw=0.9, alpha=0.35, zorder=4)
        trail_lines.append(ln)

    # Líneas de entrelazamiento (se recrean cada frame)
    ent_objs = []

    # ── Panel convergencia ───────────────────────────────────────────────────
    ax_cv.set_title('Convergencia', color=_TEXT, fontsize=9, pad=4)
    ax_cv.set_xlabel('Iteración', color='#8888aa', fontsize=7)
    ax_cv.set_ylabel('f_min', color='#8888aa', fontsize=7)
    ax_cv.tick_params(colors='#555566', labelsize=6)
    ax_cv.grid(True, color=_GRID, lw=0.5)

    bv = sim.h_best
    ym = (max(bv) - min(bv)) * 0.1 + 0.1
    ax_cv.set_xlim(0, num_iters)
    ax_cv.set_ylim(min(bv) - ym, max(bv) + ym)

    cv_line, = ax_cv.plot([], [], color=_CYAN, lw=1.5)
    cv_dot,  = ax_cv.plot([], [], 'o', color=_ORANGE, ms=5, zorder=5)

    # ── Panel proyección Bloch ───────────────────────────────────────────────
    ax_bl.set_title('Estado cuántico · plano ecuatorial Bloch',
                    color=_TEXT, fontsize=8, pad=4)
    ax_bl.set_xlim(-1.18, 1.18); ax_bl.set_ylim(-1.18, 1.18)
    ax_bl.set_aspect('equal')
    ax_bl.tick_params(colors='#555566', labelsize=6)
    ax_bl.grid(True, color=_GRID, lw=0.5)

    # Círculo unitario (ecuador de la esfera)
    th = np.linspace(0, 2 * np.pi, 300)
    ax_bl.plot(np.cos(th), np.sin(th), color='#2a2a44', lw=1.2)
    ax_bl.axhline(0, color='#222233', lw=0.5)
    ax_bl.axvline(0, color='#222233', lw=0.5)
    ax_bl.text( 0,  1.10, '|+y⟩', color=_DIM, ha='center', va='bottom', fontsize=6)
    ax_bl.text( 1.10,  0, '|+x⟩', color=_DIM, ha='left', va='center', fontsize=6)
    ax_bl.text( 0, -1.14, 'sin(α)cos(β)', color=_DIM, ha='center', fontsize=5.5)

    # Círculo de umbral de entrelazamiento (guía visual)
    r_thr = sim.ENTANGLEMENT_THRESHOLD ** 0.5
    ax_bl.plot(r_thr * np.cos(th), r_thr * np.sin(th),
               color=_GOLD + '55', lw=0.8, ls='--')
    ax_bl.text(r_thr * 0.68, r_thr * 0.68, 'umbral F',
               color=_GOLD + '88', fontsize=5, ha='center')

    bl_scat = ax_bl.scatter([], [], s=65, zorder=5)

    # ── Panel leyenda / info ─────────────────────────────────────────────────
    ax_lg.axis('off')
    ax_lg.set_title('Leyenda', color=_TEXT, fontsize=9, pad=4)

    items = [
        ('●', cmap_q(0.0), 'α ≈ 0   conservador\n  P(túnel) ≈ 0'),
        ('●', cmap_q(0.5), 'α = π/2  indeciso\n  P(túnel) = 0.5'),
        ('●', cmap_q(1.0), 'α ≈ π   temerario\n  P(túnel) ≈ 1'),
        ('━', _GOLD,       'Entrelazamiento\n  (F > umbral)'),
        ('●', 'white',     'Evento de túnel\n  cuántico'),
    ]
    for k, (sym, col, txt) in enumerate(items):
        y = 0.92 - k * 0.18
        ax_lg.text(0.04, y, sym, color=col, fontsize=15,
                   transform=ax_lg.transAxes, va='top')
        ax_lg.text(0.18, y, txt, color=_TEXT, fontsize=7.5,
                   transform=ax_lg.transAxes, va='top',
                   linespacing=1.4)

    ax_lg.text(
        0.04, 0.05,
        'Canicas = esferas de Bloch\n'
        'Al rodar, su estado cuántico\n'
        'evoluciona con la curvatura\n'
        'de la hipersuperficie',
        color='#555577', fontsize=6.5,
        transform=ax_lg.transAxes, va='bottom', style='italic'
    )

    # ── Función de actualización ──────────────────────────────────────────────

    def update(frame):
        nonlocal ent_objs

        pos      = sim.h_pos[frame]
        alphas   = sim.h_alpha[frame]
        betas    = sim.h_beta[frame]
        objs_f   = sim.h_obj[frame]
        best     = sim.h_best[frame]
        F        = sim.h_fidelity[frame]
        entangl  = sim.h_entangled[frame]
        tunneled = sim.h_tunneled[frame]

        zs = np.array([rugose(pos[i, 0], pos[i, 1]) for i in range(num_agents)])

        # Color ← estado cuántico α
        norm_a = alphas / np.pi
        colors = cmap_q(norm_a)

        # Tamaño ← calidad de la posición (mejor objetivo = canica más grande)
        spread = objs_f.max() - objs_f.min() + 1e-9
        norm_obj = (objs_f - objs_f.min()) / spread
        sizes = 55 + 160 * (1 - norm_obj)

        # Flash blanco en eventos de túnel
        for i in tunneled:
            colors[i] = np.array([1.0, 1.0, 1.0, 1.0])
            sizes[i] = 380

        # Actualizar scatter 3D
        scat._offsets3d = (pos[:, 0], pos[:, 1], zs + 0.08)
        scat.set_color(colors)
        scat.set_sizes(sizes)

        # Rastros de posiciones anteriores
        start = max(0, frame - TRAIL)
        for i, ln in enumerate(trail_lines):
            tx = [sim.h_pos[t][i, 0] for t in range(start, frame + 1)]
            ty = [sim.h_pos[t][i, 1] for t in range(start, frame + 1)]
            tz = [rugose(sim.h_pos[t][i, 0], sim.h_pos[t][i, 1]) + 0.04
                  for t in range(start, frame + 1)]
            ln.set_data(tx, ty)
            ln.set_3d_properties(tz)
            # Color del rastro = mismo hue que la canica pero muy tenue
            rc = list(cmap_q(norm_a[i])[:3]) + [0.22]
            ln.set_color(rc)

        # Eliminar y redibujar líneas de entrelazamiento
        for ln in ent_objs:
            ln.remove()
        ent_objs = []
        for (i, j) in entangl:
            fid = F[i, j]
            vis_alpha = (fid - sim.ENTANGLEMENT_THRESHOLD) / (
                1.0 - sim.ENTANGLEMENT_THRESHOLD + 1e-9)
            ln, = ax3d.plot(
                [pos[i, 0], pos[j, 0]],
                [pos[i, 1], pos[j, 1]],
                [zs[i] + 0.08, zs[j] + 0.08],
                color=_GOLD,
                alpha=float(np.clip(vis_alpha * 0.9, 0.1, 0.9)),
                lw=0.8 + 2.0 * fid,
                zorder=6
            )
            ent_objs.append(ln)

        # Convergencia
        cv_line.set_data(range(frame + 1), sim.h_best[:frame + 1])
        cv_dot.set_data([frame], [best])

        # Proyección esfera de Bloch: (sin α cos β, sin α sin β)
        bx = np.sin(alphas) * np.cos(betas)
        by = np.sin(alphas) * np.sin(betas)
        bl_scat.set_offsets(np.column_stack([bx, by]))
        bl_scat.set_color(colors)

        # Textos informativos
        iter_txt.set_text(f'Iteración  {frame + 1} / {num_iters}')
        best_txt.set_text(
            f'f_min = {best:.4f}\n'
            f'enlaces: {len(entangl)}   '
            f'túneles: {len(tunneled)}'
        )

        return [scat, cv_line, cv_dot, bl_scat] + ent_objs + trail_lines

    # ── Ejecutar ──────────────────────────────────────────────────────────────

    anim = FuncAnimation(
        fig, update,
        frames=num_iters,
        interval=90,   # ms entre frames
        blit=False      # blit=True no funciona bien con Axes3D
    )

    if save_gif:
        fname = 'simulation.gif'
        print(f"Guardando '{fname}' (puede tardar ~1–2 min)...")
        anim.save(fname, writer='pillow', fps=12, dpi=110)
        print(f"Listo: {fname}")
    else:
        plt.show()

    return anim


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == '__main__':
    ap = argparse.ArgumentParser(
        description='QIMAD — simulación visual de canicas cuánticas')
    ap.add_argument('--save',   action='store_true',
                    help='Guardar como simulation.gif (requiere Pillow)')
    ap.add_argument('--agents', type=int, default=8,
                    help='Número de canicas / agentes (default: 8)')
    ap.add_argument('--iters',  type=int, default=120,
                    help='Iteraciones de optimización (default: 120)')
    ap.add_argument('--seed',   type=int, default=42,
                    help='Semilla aleatoria (default: 42)')
    args = ap.parse_args()

    run_simulation(
        num_agents=args.agents,
        num_iters=args.iters,
        seed=args.seed,
        save_gif=args.save,
    )
