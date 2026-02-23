import numpy as np
import imageio.v2 as imageio

def lap(Z):
    return (-4*Z + np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))

def neigh_mean(Z):
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1)) / 4.0

def run(seed=3, H=160, W=160, T=800, capture_every=6, out_gif="pixel_cell_m1.gif"):
    rng = np.random.default_rng(seed)

    # Fields
    A = (0.02*rng.standard_normal((H, W))).astype(np.float32)          # activity (fast)
    M = (0.02*rng.standard_normal((H, W))).astype(np.float32)          # memory (slow)
    E = (0.65 + 0.05*rng.standard_normal((H, W))).astype(np.float32)   # energy
    C = (0.25 + 0.02*rng.standard_normal((H, W))).astype(np.float32)   # connectivity

    # Embryo seed
    yy, xx = np.mgrid[0:H, 0:W]
    cx, cy = W//2, H//2
    embryo = ((xx-cx)**2 + (yy-cy)**2) < (min(H, W)*0.12)**2
    A[embryo] += 0.8
    E[embryo] += 0.25

    # Parameters
    dt = 0.55
    leak = 0.025
    mem_alpha = 0.04

    energy_in = 0.012
    energy_diff = 0.10
    act_cost = 0.018
    learn_cost = 0.010

    plastic_lr = 0.030
    prune_rate = 0.0025
    C_min, C_max = 0.02, 0.95

    noise = 0.010

    prev_err = np.abs(A - neigh_mean(A)).astype(np.float32)

    frames = []
    for t in range(T):
        # (1) Energy diffuses + refills
        E += energy_diff * lap(E) + energy_in
        E = np.clip(E, 0.0, 1.25)

        # (2) Sense neighbors
        N = neigh_mean(A)

        # (3) Activity update (gated by energy)
        gain = np.clip(E, 0.0, 1.0)
        drive = C * (N - A) + 0.35*M
        A = A + dt * (np.tanh(drive) * gain - leak*A) + noise*rng.standard_normal((H, W)).astype(np.float32)
        A = np.clip(A, -1.2, 1.2)

        # (4) Slow memory
        M = (1-mem_alpha)*M + mem_alpha*A
        M = np.clip(M, -1.0, 1.0)

        # Curiosity proxy: local predictability improves
        err = np.abs(A - neigh_mean(A))
        progress = np.clip(prev_err - err, 0.0, 1.0)
        prev_err = err

        # (5) Plasticity: strengthen correlated connections, prune others
        corr = A * N
        gate = (0.25 + 2.5*progress) * np.clip(E, 0.0, 1.0)
        dC = plastic_lr * gate * np.tanh(corr) - prune_rate*(1.0-gate)
        C = np.clip(C + dC, C_min, C_max)

        # (6) Energy cost
        E -= act_cost*np.abs(A) + learn_cost*np.abs(dC)
        E = np.clip(E, 0.0, 1.25)

        # Render RGB
        if t % capture_every == 0:
            R = (A + 1.2) / 2.4
            G = (M + 1.0) / 2.0
            B = np.clip(0.55*C + 0.45*np.clip(E/1.25, 0, 1), 0, 1)
            rgb = np.clip(np.stack([R, G, B], axis=-1), 0, 1)
            frames.append((rgb*255).astype(np.uint8))

    imageio.mimsave(out_gif, frames, duration=0.06)
    print("Saved:", out_gif)

if __name__ == "__main__":
    run()