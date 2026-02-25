import numpy as np
import imageio.v2 as imageio

def lap(Z):
    return (-4*Z + np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1))

def neigh_mean(Z):
    return (np.roll(Z,1,0)+np.roll(Z,-1,0)+np.roll(Z,1,1)+np.roll(Z,-1,1)) / 4.0

def downsample(Z, factor):
    return Z.reshape(Z.shape[0]//factor, factor, Z.shape[1]//factor, factor).mean(axis=(1,3))

def upsample(Z, factor):
    return np.repeat(np.repeat(Z, factor, axis=0), factor, axis=1)

def run(seed=3, H=160, W=160, T=1800, capture_every=6, out_gif="pixel_cell_m4_hierarchy.gif"):
    rng = np.random.default_rng(seed)

    # --- Micro Layer ---
    A = 0.02*rng.standard_normal((H,W)).astype(np.float32)
    M = 0.02*rng.standard_normal((H,W)).astype(np.float32)
    E = (0.65+0.05*rng.standard_normal((H,W))).astype(np.float32)
    C = (0.25+0.02*rng.standard_normal((H,W))).astype(np.float32)
    P = 0.02*rng.standard_normal((H,W)).astype(np.float32)

    # --- Macro Layer (slower brain) ---
    factor = 4
    H2, W2 = H//factor, W//factor
    A2 = np.zeros((H2,W2), dtype=np.float32)
    M2 = np.zeros((H2,W2), dtype=np.float32)

    # Embryo
    yy, xx = np.mgrid[0:H,0:W]
    cx, cy = W//2, H//2
    embryo = ((xx-cx)**2+(yy-cy)**2)<(min(H,W)*0.12)**2
    A[embryo]+=0.8
    E[embryo]+=0.25

    # Long-range links
    num_links = 300
    links_src = rng.integers(0,H,size=(num_links,2))
    links_dst = rng.integers(0,H,size=(num_links,2))

    # Parameters
    dt=0.55
    leak=0.025
    mem_alpha=0.04
    macro_alpha=0.01
    noise=0.010

    energy_in=0.010
    energy_diff=0.10
    act_cost=0.016
    learn_cost=0.010

    plastic_lr=0.028
    prune_rate=0.0025
    C_min,C_max=0.02,0.95

    pred_lr=0.12
    curiosity_boost=0.06

    short_alpha=0.2
    long_alpha=0.02

    err_short=np.zeros((H,W),dtype=np.float32)
    err_long=np.zeros((H,W),dtype=np.float32)

    frames=[]

    for t in range(T):

        # Energy
        E+=energy_diff*lap(E)+energy_in
        E=np.clip(E,0,1.25)

        # Micro neighbor
        N=neigh_mean(A)

        # Long-range signal
        LR=np.zeros_like(A)
        for i in range(num_links):
            sy,sx=links_src[i]
            dy,dx=links_dst[i]
            LR[dy,dx]+=0.1*A[sy,sx]

        # Macro feedback
        A2=(1-macro_alpha)*A2+macro_alpha*downsample(A,factor)
        macro_up=upsample(A2,factor)

        # Activity update
        gain=np.clip(E,0,1)
        drive=C*(N-A)+0.35*M+0.2*macro_up+LR
        A_next=A+dt*(np.tanh(drive)*gain-leak*A)+noise*rng.standard_normal((H,W))
        A_next=np.clip(A_next,-1.2,1.2)

        # Prediction error
        err=np.abs(P-A_next)
        err_short=(1-short_alpha)*err_short+short_alpha*err
        err_long=(1-long_alpha)*err_long+long_alpha*err
        progress=np.clip(err_long-err_short,0,1)

        # Prediction update
        P+=pred_lr*(A_next-P)*gain
        P=np.clip(P,-1.2,1.2)

        A=A_next

        # Memory
        M=(1-mem_alpha)*M+mem_alpha*A
        M=np.clip(M,-1,1)

        # Plasticity
        corr=A*neigh_mean(A)
        gate=(0.25+3*progress)*gain
        dC=plastic_lr*gate*np.tanh(corr)-prune_rate*(1-gate)
        C=np.clip(C+dC,C_min,C_max)

        # Energy dynamics
        E+=curiosity_boost*progress
        E-=act_cost*np.abs(A)+learn_cost*np.abs(dC)
        E=np.clip(E,0,1.25)

        if t%capture_every==0:
            R=(A+1.2)/2.4
            G=(M+1)/2
            B=np.clip(0.5*C+0.4*progress+0.3*np.clip(E/1.25,0,1),0,1)
            rgb=np.clip(np.stack([R,G,B],axis=-1),0,1)
            frames.append((rgb*255).astype(np.uint8))

    imageio.mimsave(out_gif,frames,duration=0.06)
    print("Saved:",out_gif)

if __name__=="__main__":
    run()