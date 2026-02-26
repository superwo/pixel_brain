import matplotlib
matplotlib.use("TkAgg")  # or "QtAgg"
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ==========================
# CONFIG
# ==========================

H, W = 256, 256  # Internal simulation grid
GENES = 8
UPSCALE = 4  # 256 -> 1024 rendering
RENDER_EVERY = 10  # Render every N steps

DT = 0.1
ENERGY_DIFFUSION = 0.15
BASE_ENERGY = 0.02
GRADIENT_STRENGTH = 0.08

METABOLIC_COST = 0.015
MAINTENANCE_COST = 0.005

DAMAGE_STRESS_FACTOR = 0.02
DAMAGE_REPAIR_RATE = 0.01

GROWTH_THRESHOLD = 1.2
DEATH_THRESHOLD = 0.5

# ==========================
# INITIAL STATE
# ==========================

rng = np.random.default_rng(42)

# Gene expression: (H, W, GENES)
X = np.zeros((H, W, GENES), dtype=np.float32)

# Local energy and damage
E = np.zeros((H, W), dtype=np.float32)
D = np.zeros((H, W), dtype=np.float32)

# Alive mask
alive = np.zeros((H, W), dtype=bool)

# Start with a small seed cluster in center
alive[H // 2 - 3 : H // 2 + 3, W // 2 - 3 : W // 2 + 3] = True
X[alive] = rng.normal(0, 0.1, (alive.sum(), GENES))
E[alive] = 0.8

# ==========================
# DNA (shared by all cells)
# ==========================

# Sparse regulatory matrix
W_reg = rng.normal(0, 0.5, (GENES, GENES))
mask = rng.random((GENES, GENES)) < 0.5
W_reg *= mask

survival_weights = rng.normal(0.5, 0.2, GENES)
death_weights = rng.normal(0.5, 0.2, GENES)

# ==========================
# Helper Functions
# ==========================


def laplacian(Z):
    return (
        -4 * Z
        + np.roll(Z, 1, 0)
        + np.roll(Z, -1, 0)
        + np.roll(Z, 1, 1)
        + np.roll(Z, -1, 1)
    )


def energy_gradient():
    y = np.linspace(0, 1, H)
    grad = BASE_ENERGY + GRADIENT_STRENGTH * (1 - y)
    return np.tile(grad[:, None], (1, W))


energy_input = energy_gradient()

# ==========================
# UPDATE STEP
# ==========================


def step():
    global X, E, D, alive

    # ----------------------
    # Energy input + diffusion
    # ----------------------
    E += energy_input
    E += ENERGY_DIFFUSION * laplacian(E)
    E = np.clip(E, 0, 2.0)

    # ----------------------
    # Gene regulation
    # ----------------------
    for g in range(GENES):
        reg_input = np.tensordot(X, W_reg[:, g], axes=([2], [0]))
        X[..., g] = np.tanh(reg_input)

    X[~alive] = 0

    # ----------------------
    # Metabolism
    # ----------------------
    activity_level = np.linalg.norm(X, axis=2)
    E -= METABOLIC_COST * activity_level
    E -= MAINTENANCE_COST

    # ----------------------
    # Damage dynamics
    # ----------------------
    stress = np.maximum(0, 0.3 - E)
    D += DAMAGE_STRESS_FACTOR * stress
    D -= DAMAGE_REPAIR_RATE * E
    D = np.clip(D, 0, 2.0)

    # ----------------------
    # Survival decision
    # ----------------------
    survival_score = np.tensordot(X, survival_weights, axes=([2], [0])) + E
    death_score = np.tensordot(X, death_weights, axes=([2], [0])) + D

    die = (death_score > survival_score) & alive
    alive[die] = False
    X[die] = 0
    E[die] = 0
    D[die] = 0

    # ----------------------
    # Slow growth
    # ----------------------
    grow_candidates = (E > GROWTH_THRESHOLD) & alive

    for i, j in zip(*np.where(grow_candidates)):
        neighbors = [
            ((i + 1) % H, j),
            ((i - 1) % H, j),
            (i, (j + 1) % W),
            (i, (j - 1) % W),
        ]
        rng.shuffle(neighbors)
        for ni, nj in neighbors:
            if not alive[ni, nj]:
                alive[ni, nj] = True
                X[ni, nj] = rng.normal(0, 0.05, GENES)
                E[ni, nj] = 0.5
                break


# ==========================
# VISUALIZATION
# ==========================


def render():
    # Gene 0 -> Red
    # Gene 1 -> Blue
    # Energy -> Brightness

    img = np.zeros((H, W, 3), dtype=np.float32)

    img[..., 0] = np.clip(X[..., 0], -1, 1) * 0.5 + 0.5
    img[..., 2] = np.clip(X[..., 1], -1, 1) * 0.5 + 0.5

    brightness = np.clip(E / 1.5, 0, 1)
    img *= brightness[..., None]

    img[~alive] = 0

    img = np.kron(img, np.ones((UPSCALE, UPSCALE, 1)))

    return img


# ==========================
# MAIN LOOP
# ==========================

fig = plt.figure()
im = plt.imshow(render())
plt.axis("off")


def update(frame):
    for _ in range(RENDER_EVERY):
        step()
    im.set_array(render())
    return [im]


ani = FuncAnimation(fig, update, interval=50)
plt.show()
