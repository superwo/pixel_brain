import matplotlib
matplotlib.use("TkAgg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

# ==========================
# CONFIG
# ==========================

H, W = 256, 256
GENES = 8
UPSCALE = 4
RENDER_EVERY = 10

BASE_ENERGY = 0.02
ENERGY_DIFFUSION = 0.05
GRADIENT_STRENGTH = 0.12

INHIBITION_STRENGTH = 0.15

METABOLIC_COST = 0.015
MAINTENANCE_COST = 0.005

DAMAGE_STRESS_FACTOR = 0.02
DAMAGE_REPAIR_RATE = 0.01

GROWTH_THRESHOLD = 1.2

# ==========================
# TEMPORAL TASK
# ==========================

DELAY = 25

time_step = 0
input_history = []

# ==========================
# INITIAL STATE
# ==========================

rng = np.random.default_rng(42)

X = np.zeros((H, W, GENES), dtype=np.float32)
E = np.zeros((H, W), dtype=np.float32)
D = np.zeros((H, W), dtype=np.float32)
alive = np.zeros((H, W), dtype=bool)

alive[H//2-3:H//2+3, W//2-3:W//2+3] = True
X[alive] = rng.normal(0, 0.1, (alive.sum(), GENES))
E[alive] = 0.8

# ==========================
# DNA
# ==========================

W_reg = rng.normal(0, 0.5, (GENES, GENES))
mask = rng.random((GENES, GENES)) < 0.5
W_reg *= mask

survival_weights = rng.normal(0.5, 0.2, GENES)
death_weights = rng.normal(0.5, 0.2, GENES)

# ==========================
# HELPERS
# ==========================

def neighbor_mean(Z):
    return (
        np.roll(Z,1,0) + np.roll(Z,-1,0) +
        np.roll(Z,1,1) + np.roll(Z,-1,1)
    ) / 4.0

def laplacian(Z):
    return (
        -4*Z +
        np.roll(Z,1,0) + np.roll(Z,-1,0) +
        np.roll(Z,1,1) + np.roll(Z,-1,1)
    )

def energy_gradient():
    y = np.linspace(0, 1, H)
    grad = BASE_ENERGY + GRADIENT_STRENGTH * (1 - y)
    return np.tile(grad[:,None], (1,W))

energy_input = energy_gradient()

# ==========================
# STEP FUNCTION
# ==========================

def step():
    global time_step, input_history
    global X, E, D, alive

    # ----------------------
    # Generate non-periodic XOR input
    # ----------------------
    if len(input_history) < 2:
        current_input = rng.integers(0,2)
    else:
        current_input = input_history[-1] ^ input_history[-2]

    input_history.append(current_input)

    # ----------------------
    # Energy input + diffusion
    # ----------------------
    E += energy_input

    # Remove baseline energy from bottom region
    E[int(H*0.75):H, :] -= BASE_ENERGY

    E += ENERGY_DIFFUSION * laplacian(E)
    E = np.clip(E, 0, 2.0)

    # ----------------------
    # Gene regulation
    # ----------------------
    inhibition_field = neighbor_mean(X[...,2])

    new_X = np.zeros_like(X)

    for g in range(GENES):
        reg_input = np.tensordot(X, W_reg[:, g], axes=([2],[0]))

        # Inject sensory signal into regulation (not directly into X)
        if g == 0:
            reg_input[0:8, :] += current_input * 0.5

        if g != 2:
            reg_input -= INHIBITION_STRENGTH * inhibition_field

        new_X[..., g] = np.tanh(reg_input)

    X = new_X
    X[~alive] = 0

    # ----------------------
    # Metabolism
    # ----------------------
    activity_level = np.linalg.norm(X, axis=2)
    E -= METABOLIC_COST * activity_level
    E -= MAINTENANCE_COST

    # ----------------------
    # Damage
    # ----------------------
    stress = np.maximum(0, 0.3 - E)
    D += DAMAGE_STRESS_FACTOR * stress
    D -= DAMAGE_REPAIR_RATE * E
    D = np.clip(D, 0, 2.0)

    # ----------------------
    # Survival
    # ----------------------
    survival_score = np.tensordot(X, survival_weights, axes=([2],[0])) + E
    death_score = np.tensordot(X, death_weights, axes=([2],[0])) + D

    die = (death_score > survival_score) & alive
    alive[die] = False
    X[die] = 0
    E[die] = 0
    D[die] = 0

    # ----------------------
    # Compute Output (bottom band)
    # ----------------------
    bottom_band = X[H-8:H, :, 3]
    output_value = np.mean(bottom_band)

    if len(input_history) > DELAY:
        target = input_history[-DELAY]
    else:
        target = 0
    bottom_band = X[H-8:H, :, 3]
    output_value = np.mean(bottom_band)

    if len(input_history) > DELAY:
        target = input_history[-DELAY]
    else:
        target = 0
    bottom_band = X[H-8:H, :, 3]
    output_value = np.mean(bottom_band)

    if len(input_history) > DELAY:
        target = input_history[-DELAY]
    else:
        target = 0

    # ----------------------
    # Competitive Reward
    # ----------------------
    mid = W // 2
    error = output_value - target

    reward_strength = 0.05

    # Signed reinforcement
    E[H-8:H, :mid] += reward_strength * (target - output_value)
    E[H-8:H, mid:] -= reward_strength * (target - output_value)

    # ----------------------
    # Growth
    # ----------------------
    grow_candidates = (E > GROWTH_THRESHOLD) & alive

    for i, j in zip(*np.where(grow_candidates)):
        neighbors = [
            ((i+1)%H, j),
            ((i-1)%H, j),
            (i, (j+1)%W),
            (i, (j-1)%W),
        ]
        rng.shuffle(neighbors)
        for ni, nj in neighbors:
            if not alive[ni, nj]:
                alive[ni, nj] = True
                X[ni, nj] = rng.normal(0, 0.05, GENES)
                E[ni, nj] = 0.5
                break

    time_step += 1

# ==========================
# VISUALIZATION
# ==========================

def render():
    img = np.zeros((H, W, 3), dtype=np.float32)

    img[...,0] = np.clip(X[...,0], -1,1)*0.5 + 0.5
    img[...,2] = np.clip(X[...,1], -1,1)*0.5 + 0.5

    brightness = np.clip(E / 1.5, 0, 1)
    img *= brightness[...,None]

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