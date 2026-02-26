import numpy as np

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

# ---------------- Main ----------------
def run(seed=3, H=120, W=120, T=4000):

    rng = np.random.default_rng(seed)

    # ===== Core Fields =====
    A = 0.02*rng.standard_normal((H,W)).astype(np.float32)
    M = np.zeros((H,W),dtype=np.float32)
    B = np.zeros((H,W),dtype=np.float32)  # identity field

    E = (0.6+0.05*rng.standard_normal((H,W))).astype(np.float32)
    C = (0.25+0.02*rng.standard_normal((H,W))).astype(np.float32)

    factor = 6
    A_macro = np.zeros((H//factor,W//factor),dtype=np.float32)

    # ===== Sequence Generator (Noisy 3-step rule) =====
    def generate_bit(prev1, prev2, prev3):
        base = (prev1 ^ prev2 ^ prev3)
        noise = rng.random() < 0.05
        return base ^ noise

    history = [1,0,1]

    def get_bit(t):
        nonlocal history
        if t < 3:
            return history[t]
        new_bit = generate_bit(history[-1], history[-2], history[-3])
        history.append(new_bit)
        return new_bit

    # ===== Readout =====
    w = rng.normal()*0.1
    b_out = 0.0
    lr = 0.01

    # ===== Parameters =====
    dt=0.5
    leak=0.02
    macro_alpha=0.01
    noise_amp=0.006

    energy_in=0.01
    energy_diff=0.1

    lambda_b = 0.5  # identity influence
    eta = 0.002     # identity accumulation
    decay = 0.0003  # identity decay

    cycle_length = 40
    calibration_steps = 8

    correct = 0
    total = 0

    for t in range(T):

        in_calibration = (t % cycle_length) < calibration_steps

        # ===== Input =====
        x_t = get_bit(t)
        x_next = get_bit(t+1)

        input_region = np.zeros((H,W), dtype=np.float32)
        input_region[5:15, 5:15] = float(x_t)

        if in_calibration:
            O = input_region + 0.05*rng.standard_normal((H,W))
        else:
            O = input_region * 0.2

        # ===== Energy =====
        E += energy_diff*lap(E)+energy_in
        E = np.clip(E,0,1.2)
        gain = np.clip(E,0,1)

        # ===== Macro =====
        A_macro = (1-macro_alpha)*A_macro + macro_alpha*downsample(A,factor)
        macro_up = upsample(A_macro,factor)

        # ===== Core Drive with Identity Reinforcement =====
        N = neigh_mean(A)
        local = A - N

        drive = C*(N-A) + 0.3*M + 0.25*macro_up + 0.3*O + lambda_b * B

        drive += 0.2 * local
        A = A + dt*(np.tanh(drive)*gain - leak*A)
        A += noise_amp*rng.standard_normal((H,W))
        A = np.clip(A,-1.2,1.2)

        # ===== Slow Memory Trace =====
        M = 0.98*M + 0.02*A

        # ===== Identity Reinforcement =====
        B += eta * A
        B -= decay * B
        B = np.clip(B, -2.0, 2.0)
        # Subtract neighborhood mean (local competition)
        B -= 0.1 * (neigh_mean(B) - B)

        # ===== Readout =====
        read_region = A[H-15:H-5, W-15:W-5]
        y_internal = np.mean(read_region)

        y_pred = w*y_internal + b_out
        y_pred_sigmoid = 1/(1+np.exp(-y_pred))

        error = y_pred_sigmoid - x_next
        grad = error * y_pred_sigmoid*(1-y_pred_sigmoid)

        w -= lr * grad * y_internal
        b_out -= lr * grad

        prediction = 1 if y_pred_sigmoid > 0.5 else 0

        if prediction == x_next:
            correct += 1
        total += 1

        if t % 500 == 0 and t > 0:
            print(f"Step {t}, Accuracy: {correct/total:.3f}")
            correct = 0
            total = 0

    print("Finished.")

if __name__ == "__main__":
    run()