import numpy as np
import imageio.v2 as imageio
from collections import deque

# ---------------- Utilities ----------------
def lap(Z):
    return (-4*Z + np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))

def neigh_mean(Z):
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))/4.0

def downsample(Z, factor):
    return Z.reshape(Z.shape[0]//factor, factor,
                     Z.shape[1]//factor, factor).mean(axis=(1,3))

def upsample(Z, factor):
    return np.repeat(np.repeat(Z, factor, axis=0), factor, axis=1)

# ---------------- World Blob ----------------
class Blob:
    def __init__(self, H, W, rng):
        self.H = H
        self.W = W
        self.x = rng.uniform(0, W)
        self.y = rng.uniform(0, H)
        self.vx = rng.uniform(-0.7, 0.7)
        self.vy = rng.uniform(-0.7, 0.7)
        self.size = rng.uniform(6, 12)
        self.intensity = rng.uniform(0.6, 1.2)

    def update(self):
        self.x += self.vx
        self.y += self.vy
        if self.x < 0 or self.x >= self.W: self.vx *= -1
        if self.y < 0 or self.y >= self.H: self.vy *= -1

    def render(self, gx, gy):
        return self.intensity * np.exp(
            -((gx-self.x)**2+(gy-self.y)**2)/(2*self.size**2)
        )

# ---------------- Main ----------------
def run(seed=3, H=160, W=160, T=3000,
        capture_every=6,
        out_gif="pixel_cell_m8_continuous.gif"):

    rng = np.random.default_rng(seed)

    # -------- Brain --------
    A = 0.02*rng.standard_normal((H,W)).astype(np.float32)
    M = np.zeros((H,W),dtype=np.float32)
    E = (0.65+0.05*rng.standard_normal((H,W))).astype(np.float32)
    C = (0.25+0.02*rng.standard_normal((H,W))).astype(np.float32)
    P = np.zeros((H,W),dtype=np.float32)

    factor = 8
    A2 = np.zeros((H//factor,W//factor),dtype=np.float32)

    # -------- World --------
    W_wave = 0.1*rng.standard_normal((H,W)).astype(np.float32)
    blobs = [Blob(H,W,rng) for _ in range(4)]
    gy,gx = np.mgrid[0:H,0:W]

    # -------- Future Buffer --------
    k_future = 5
    world_buffer = deque(maxlen=k_future+1)

    # -------- Timing (20/80) --------
    cycle_length = 50
    calibration_steps = 10

    # -------- Parameters --------
    dt=0.5
    leak=0.02
    mem_alpha=0.04
    macro_alpha=0.01
    noise=0.01

    energy_in=0.01
    energy_diff=0.1
    curiosity_boost=0.08

    plastic_lr=0.025
    prune_rate=0.003
    pred_lr=0.2

    short_alpha=0.25
    long_alpha=0.02

    err_short=np.zeros((H,W),dtype=np.float32)
    err_long=np.zeros((H,W),dtype=np.float32)
    progress=np.zeros((H,W),dtype=np.float32)

    retina_size=40
    rx, ry = 0, 0
    rvx, rvy = 0.5, 0.3

    frames=[]

    for t in range(T):

        in_calibration = (t % cycle_length) < calibration_steps

        # ---------- WORLD ----------
        W_wave += 0.15*lap(W_wave)
        W_wave += 0.01*np.sin(t*0.02)

        W_objects = np.zeros((H,W),dtype=np.float32)
        for blob in blobs:
            blob.update()
            W_objects += blob.render(gx,gy)

        W_total = np.clip(W_wave + W_objects, -2, 2)
        world_buffer.append(W_total.copy())

        # ---------- ACTIVE RETINA ----------
        grad_y, grad_x = np.gradient(err_short)
        rvx += 0.15 * np.mean(grad_x)
        rvy += 0.15 * np.mean(grad_y)

        rx = int((rx + rvx) % (W-retina_size))
        ry = int((ry + rvy) % (H-retina_size))

        mask = np.zeros((H,W),dtype=np.float32)
        mask[ry:ry+retina_size, rx:rx+retina_size] = 1.0

        # Perception only during calibration
        if in_calibration:
            O = W_total * mask + 0.05*rng.standard_normal((H,W))
        else:
            O = np.zeros((H,W), dtype=np.float32)

        # ---------- BRAIN DYNAMICS ----------
        E += energy_diff*lap(E)+energy_in
        E = np.clip(E,0,1.2)

        N = neigh_mean(A)

        A2 = (1-macro_alpha)*A2 + macro_alpha*downsample(A,factor)
        macro_up = upsample(A2,factor)

        gain = np.clip(E,0,1)

        drive = C*(N-A) + 0.3*M + 0.25*macro_up + 0.3*O
        A_next = A + dt*(np.tanh(drive)*gain - leak*A)
        A_next += noise*rng.standard_normal((H,W))
        A_next = np.clip(A_next,-1.2,1.2)

        A = A_next
        M = (1-mem_alpha)*M + mem_alpha*A

        # ---------- FUTURE PREDICTION ----------
        if len(world_buffer) > k_future and in_calibration:
            target = world_buffer[0]
            err = np.abs(P - target)

            err_short = (1-short_alpha)*err_short + short_alpha*err
            err_long  = (1-long_alpha)*err_long  + long_alpha*err
            progress  = np.clip(err_long-err_short,0,1)

            P += pred_lr*(target-P)*gain
            P = np.clip(P,-2,2)
        else:
            progress *= 0.98  # decay during imagination

        # ---------- PLASTICITY ----------
        corr = A*neigh_mean(A)
        gate = (0.3+3*progress)*gain
        dC = plastic_lr*gate*np.tanh(corr)-prune_rate*(1-gate)
        C = np.clip(C+dC,0.02,0.9)

        # ---------- ENERGY ----------
        E += curiosity_boost*progress
        E -= 0.02*np.abs(A)+0.015*np.abs(dC)
        E = np.clip(E,0,1.2)

        # ---------- RENDER ----------
        if t%capture_every==0:
            R = (W_total+2)/4
            G = (A+1.2)/2.4
            B = np.clip(progress+0.3*C,0,1)
            rgb = np.clip(np.stack([R,G,B],axis=-1),0,1)
            frames.append((rgb*255).astype(np.uint8))

    imageio.mimsave(out_gif,frames,duration=0.06)
    print("Saved:",out_gif)

if __name__=="__main__":
    run()