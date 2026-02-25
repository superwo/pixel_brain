import numpy as np
import imageio.v2 as imageio

def lap(Z):
    return (-4*Z + np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))

def neigh_mean(Z):
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1)) / 4.0

def run(seed=3, H=160, W=160, T=1200, capture_every=6, out_gif="pixel_cell_m2_predict.gif"):
    rng = np.random.default_rng(seed)

    # --- Fields ---
    A = (0.02*rng.standard_normal((H, W))).astype(np.float32)          # activity (fast)
    M = (0.02*rng.standard_normal((H, W))).astype(np.float32)          # memory (slow)
    E = (0.65 + 0.05*rng.standard_normal((H, W))).astype(np.float32)   # energy
    C = (0.25 + 0.02*rng.standard_normal((H, W))).astype(np.float32)   # connectivity (local coupling)

    # NEW: prediction field (tries to predict next A)
    P = (0.02*rng.standard_normal((H, W))).astype(np.float32)

    # Embryo seed
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W//2, H//2
    embryo = ((xx-cx)**2 + (yy-cy)**2) < (min(H, W)*0.12)**2
    A[embryo] += 0.8
    E[embryo] += 0.25

    # --- Dynamics params ---
    dt = 0.55
    leak = 0.025
    mem_alpha = 0.04
    noise = 0.010

    # --- Energy ---
    base_energy_in = 0.010
    energy_diff = 0.10
    act_cost = 0.016
    learn_cost = 0.010
    curiosity_energy_boost = 0.030  # extra energy where learning progress happens

    # --- Plasticity for C ---
    plastic_lr = 0.028
    prune_rate = 0.0025
    C_min, C_max = 0.02, 0.95

    # --- Prediction learning ---
    pred_lr = 0.12               # how fast P chases A_next
    curiosity_gain = 2.5         # boosts learning gate when progress is high
    err_smooth = 0.12            # smooth error to measure progress robustly

    # For curiosity = decreasing prediction error
    err_prev = np.zeros((H, W), dtype=np.float32)
    err_ema  = np.zeros((H, W), dtype=np.float32)

    frames = []
    for t in range(T):
        # (1) Energy: diffuse + baseline refill
        E += energy_diff * lap(E) + base_energy_in
        E = np.clip(E, 0.0, 1.25)

        # --- SENSE neighbors ---
        N = neigh_mean(A)

        # (2) DREAM UPDATE: activity evolves
        gain = np.clip(E, 0.0, 1.0)
        drive = C * (N - A) + 0.35*M
        A_next = A + dt * (np.tanh(drive) * gain - leak*A) + noise*rng.standard_normal((H, W)).astype(np.float32)
        A_next = np.clip(A_next, -1.2, 1.2)

        # (3) PREDICT: compare prediction to actual next activity
        err = np.abs(P - A_next).astype(np.float32)
        err_ema = (1-err_smooth)*err_ema + err_smooth*err

        # Curiosity = learning progress = error going down
        progress = np.clip(err_prev - err_ema, 0.0, 1.0)
        err_prev = err_ema.copy()

        # (4) Update prediction P to better match next activity (simple predictive learner)
        # Think: P is a "model" trying to track A's dynamics
        P = P + pred_lr * (A_next - P) * np.clip(E, 0.0, 1.0)
        P = np.clip(P, -1.2, 1.2)

        # Commit activity update
        A = A_next

        # (5) Slow memory update
        M = (1-mem_alpha)*M + mem_alpha*A
        M = np.clip(M, -1.0, 1.0)

        # (6) Plasticity: update connectivity C, gated by energy + curiosity progress
        # Local correlation proxy
        corr = A * neigh_mean(A)

        # Gate is higher where progress is high (curiosity focuses learning)
        gate = (0.25 + curiosity_gain*progress) * np.clip(E, 0.0, 1.0)

        dC = plastic_lr * gate * np.tanh(corr) - prune_rate*(1.0-gate)
        C = np.clip(C + dC, C_min, C_max)

        # (7) Curiosity affects energy: where progress happens, more "metabolic support"
        E += curiosity_energy_boost * progress
        # Costs
        E -= act_cost*np.abs(A) + learn_cost*(np.abs(dC) + 0.25*np.abs(P - A))
        E = np.clip(E, 0.0, 1.25)

        # Render RGB:
        # R = activity, G = memory, B = "structure/attention" (connectivity + curiosity)
        if t % capture_every == 0:
            R = (A + 1.2) / 2.4
            G = (M + 1.0) / 2.0
            B = np.clip(0.55*C + 0.30*np.clip(E/1.25, 0, 1) + 0.60*progress, 0, 1)
            rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 1)
            frames.append((rgb*255).astype(np.uint8))

    imageio.mimsave(out_gif, frames, duration=0.06)
    print("Saved:", out_gif)

if __name__ == "__main__":
    run()