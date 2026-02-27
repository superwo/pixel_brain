import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.animation import FuncAnimation

matplotlib.use("TkAgg")

# ==========================================================
# GRID
# ==========================================================
H, W = 256, 256
GENES = 8
UPSCALE = 4
RENDER_EVERY = 5

# ==========================================================
# FEATURE FLAGS
# ==========================================================
USE_METHYLATION = True
USE_PROTEIN_TURNOVER = True
USE_SIGNAL_FIELD = True
USE_ION_FIELD = True
USE_PLASTICITY = True
USE_STATE_MUTATION = True
USE_TASK = True

# ==========================================================
# BIO PARAMETERS
# ==========================================================
BASE_ENERGY = 0.02
GRADIENT_STRENGTH = 0.12
ENERGY_DIFFUSION = 0.05

METABOLIC_COST = 0.015
MAINTENANCE_COST = 0.005

GROWTH_THRESHOLD = 1.2
DEATH_THRESHOLD = 0.1

REWARD_STRENGTH = 0.3
DELAY = 40

# ==========================================================
# STATE
# ==========================================================
rng = np.random.default_rng(42)
time_step = 0
input_history = []
log_output = []
log_target = []

# ==========================================================
# FIELDS
# ==========================================================
X = np.zeros((H, W, GENES), dtype=np.float32)
M = np.zeros((H, W, GENES), dtype=np.float32)
E = np.zeros((H, W), dtype=np.float32)
I = np.zeros((H, W), dtype=np.float32)
S = np.zeros((H, W), dtype=np.float32)

alive = np.zeros((H, W), dtype=bool)

# Seed
alive[H // 2 - 4 : H // 2 + 4, W // 2 - 4 : W // 2 + 4] = True
X[alive] = rng.normal(0, 0.1, (alive.sum(), GENES))
E[alive] = 0.8

# ==========================================================
# DNA
# ==========================================================
W_reg = rng.normal(0, 0.5, (GENES, GENES))
survival_weights = rng.normal(0.5, 0.2, GENES)
death_weights = rng.normal(0.5, 0.2, GENES)


# ==========================================================
# HELPERS
# ==========================================================
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


# ==========================================================
# STEP
# ==========================================================
def step():
    global time_step, input_history
    global X, M, E, I, S, alive, W_reg

    # ---------------- TASK INPUT ----------------
    if USE_TASK:
        if len(input_history) < 2:
            current_input = rng.integers(0, 2)
        else:
            current_input = input_history[-1] ^ input_history[-2]
        input_history.append(current_input)
    else:
        current_input = 0

    # ---------------- ENERGY ----------------
    E += energy_input
    E[int(H * 0.75) : H, :] -= BASE_ENERGY
    E += ENERGY_DIFFUSION * laplacian(E)
    E = np.clip(E, 0, 2.0)

    # ---------------- ION FIELD ----------------
    if USE_ION_FIELD:
        I += 0.2 * laplacian(I)
        I += 0.05 * X[..., 0]
        I *= 0.98

    # ---------------- SIGNAL FIELD ----------------
    if USE_SIGNAL_FIELD:
        S += 0.05 * X[..., 1]
        S += 0.01 * laplacian(S)
        S *= 0.995

    # ---------------- GENE REGULATION ----------------
    new_X = np.zeros_like(X)

    for g in range(GENES):
        reg_input = np.tensordot(X, W_reg[:, g], axes=([2], [0]))

        if USE_TASK and g == 0:
            reg_input[0:8, :] += current_input * 0.5

        if USE_METHYLATION:
            reg_input *= 1 - M[..., g]

        if USE_SIGNAL_FIELD:
            reg_input += 0.2 * S

        if USE_ION_FIELD:
            reg_input += 0.3 * I

        target_expr = np.tanh(reg_input)

        if USE_PROTEIN_TURNOVER:
            new_X[..., g] = X[..., g] + 0.1 * (target_expr - X[..., g])
        else:
            new_X[..., g] = target_expr

    X = new_X
    X[~alive] = 0

    # ---------------- METABOLISM ----------------
    activity = np.linalg.norm(X, axis=2)
    E -= METABOLIC_COST * activity
    E -= MAINTENANCE_COST
    E = np.clip(E, 0, 2.0)

    # ---------------- EPIGENETICS ----------------
    if USE_METHYLATION:
        M += 0.001 * X
        M -= 0.0005 * E[..., None]
        M = np.clip(M, 0, 1)

    # ---------------- SURVIVAL (FIXED) ----------------
    survival_score = np.tensordot(X, survival_weights, axes=([2], [0])) + E
    death_score = np.tensordot(X, death_weights, axes=([2], [0]))

    die = ((death_score > survival_score) | (E < DEATH_THRESHOLD)) & alive

    alive[die] = False
    X[die] = 0
    M[die] = 0
    E[die] = 0

    # ---------------- OUTPUT & REWARD ----------------
    if USE_TASK:
        bottom_band = X[H - 8 : H, :, 3]
        output_value = float(np.mean(bottom_band))
        target = input_history[-DELAY] if len(input_history) > DELAY else 0

        error = target - output_value
        delta = REWARD_STRENGTH * error

        # Competitive energy shaping
        mid = W // 2
        E[H - 8 : H, :mid] += delta
        E[H - 8 : H, mid:] -= delta

        log_output.append(output_value)
        log_target.append(target)

        if time_step % 200 == 0 and len(log_target) > 100:
            corr = np.corrcoef(log_output[-100:], log_target[-100:])[0, 1]
            print("Radu")
            print(f"t={time_step} corr={corr:.3f} alive={alive.mean():.3f}")

    # ---------------- GROWTH ----------------
    grow = (E > GROWTH_THRESHOLD) & alive

    for i, j in zip(*np.where(grow)):
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
                M[ni, nj] = M[i, j]
                E[ni, nj] = 0.5
                break

    # ---------------- PLASTICITY ----------------
    if USE_PLASTICITY and USE_TASK:
        mean_expr = X.mean(axis=(0, 1))
        W_reg += 5e-5 * error * np.outer(mean_expr, mean_expr)
        W_reg = np.clip(W_reg, -2, 2)

    # ---------------- STATE MUTATION ----------------
    if USE_STATE_MUTATION:
        mutation_rate = 1e-6 + 1e-4 * M.mean()
        if rng.random() < mutation_rate:
            W_reg += rng.normal(0, 0.01, W_reg.shape)
            W_reg = np.clip(W_reg, -2, 2)

    time_step += 1


# ==========================================================
# RENDER
# ==========================================================
def render():
    img = np.zeros((H, W, 3), dtype=np.float32)

    img[..., 0] = np.clip(X[..., 0], -1, 1) * 0.5 + 0.5
    img[..., 1] = np.clip(S, -1, 1) * 0.5 + 0.5
    img[..., 2] = np.clip(I, -1, 1) * 0.5 + 0.5

    brightness = np.clip(E / 1.5, 0, 1)
    img *= brightness[..., None]

    img[~alive] = 0
    img = np.kron(img, np.ones((UPSCALE, UPSCALE, 1)))
    return img


# ==========================================================
# MAIN LOOP
# ==========================================================
fig = plt.figure()
im = plt.imshow(render())
plt.axis("off")


def update(frame):
    for _ in range(RENDER_EVERY):
        step()
    im.set_array(render())
    return [im]


ani = FuncAnimation(fig, update, interval=40)
plt.show()
