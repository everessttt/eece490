import numpy as np

# https://arxiv.org/pdf/2106.08295

NUM_STEPS = 100

# --- range methods ---

def range_minmax(
    tensor: np.ndarray,
    **kwargs,
) -> tuple[float, float]:
    return float(np.min(tensor)), float(np.max(tensor))

def range_percentile(
    tensor: np.ndarray,
    percentile: float = 99.9,
    **kwargs,
) -> tuple[float, float]:
    percentile = float(np.clip(percentile, 0, 100))
    t_min, t_max = np.percentile(tensor, [100 - percentile, percentile])
    return float(t_min), float(t_max)

def range_mse(
    tensor: np.ndarray,
    num_bits: int,
    num_steps: int = NUM_STEPS,
    **kwargs,
) -> tuple[float, float]:
    tensor = tensor.flatten()
    t_min, t_max = float(np.min(tensor)), float(np.max(tensor))

    best_mse = float("inf")
    best_min, best_max = t_min, t_max
    for i in range(1, num_steps):
        alpha = i / num_steps
        cand_min = t_min * (1 - alpha)
        cand_max = t_max * (1 - alpha)

        if cand_max <= cand_min:
            continue

        dq = np.clip(tensor, cand_min, cand_max)
        mse = float(np.mean((tensor - dq) ** 2))
        if mse < best_mse:
            best_mse = mse
            best_min, best_max = cand_min, cand_max

    return best_min, best_max

# --- quantization methods ---

def uniform_affine(
    tensor: np.ndarray,
    num_bits: int,
    range_fn,
    num_steps: int = NUM_STEPS,
    **kwargs,
) -> np.ndarray:
    t_min, t_max = range_fn(tensor, num_bits=num_bits, num_steps=num_steps)
    q_min = 0
    q_max = (1 << num_bits) - 1
    if t_max == t_min:
        return np.full_like(tensor, t_min, dtype=np.float32)
    scale = (t_max - t_min) / (q_max - q_min)
    zp = int(np.clip(np.round(-t_min / scale), q_min, q_max))
    q = np.clip(np.round(tensor / scale + zp), q_min, q_max).astype(np.int32)
    return (scale * (q - zp)).astype(np.float32)

def uniform_symmetric(
    tensor: np.ndarray,
    num_bits: int,
    range_fn,
    num_steps: int = NUM_STEPS,
    **kwargs,
) -> np.ndarray:
    t_min, t_max = range_fn(tensor, num_bits=num_bits, num_steps=num_steps)
    abs_max = max(abs(t_min), abs(t_max))
    q_min = -(1 << (num_bits - 1))
    q_max = (1 << (num_bits - 1)) - 1
    if abs_max == 0:
        return np.full_like(tensor, 0.0, dtype=np.float32)
    scale = abs_max / q_max
    q = np.clip(np.round(tensor / scale), q_min, q_max).astype(np.int32)
    return (q * scale).astype(np.float32)

def power_of_two(
    tensor: np.ndarray,
    num_bits: int,
    range_fn,
    num_steps: int = NUM_STEPS,
    **kwargs,
) -> np.ndarray:
    t_min, t_max = range_fn(tensor, num_bits=num_bits, num_steps=num_steps)
    abs_max = max(abs(t_min), abs(t_max))
    q_min = -(1 << (num_bits - 1))
    q_max = (1 << (num_bits - 1)) - 1
    if abs_max == 0:
        return np.full_like(tensor, 0.0, dtype=np.float32)
    exp = np.floor(np.log2(abs_max / q_max))
    scale = 2.0 ** exp
    q = np.clip(np.round(tensor / scale), q_min, q_max).astype(np.int32)
    return (q * scale).astype(np.float32)

def gptq(
    tensor: np.ndarray,
    num_bits: int,
    range_fn=None,
    num_steps: int = NUM_STEPS,
    calibration_input: np.ndarray = None,
    block_size: int = 128,
    damp: float = 0.01,
    **kwargs,
) -> np.ndarray:
    if calibration_input is None:
        raise ValueError("gptq requires calibration_input")

    W = tensor.astype(np.float32).copy()
    out_features, in_features = W.shape
    Q = np.zeros_like(W)

    H = 2.0 * calibration_input @ calibration_input.T
    H += damp * np.diag(H).mean() * np.eye(in_features)
    H_inv = np.linalg.inv(H)
    C = np.linalg.cholesky(H_inv).T

    q_min, q_max_val = 0, (1 << num_bits) - 1

    for i in range(0, in_features, block_size):
        j_end = min(i + block_size, in_features)
        W_block = W[:, i:j_end].copy()
        Q_block = np.zeros_like(W_block)
        E_block = np.zeros_like(W_block)

        for j in range(j_end - i):
            col = j + i
            w = W_block[:, j]
            t_min, t_max = float(np.min(w)), float(np.max(w))
            if t_max == t_min:
                Q_block[:, j] = t_min
                continue
            scale = (t_max - t_min) / (q_max_val - q_min)
            zp = int(np.clip(np.round(-t_min / scale), q_min, q_max_val))
            q = np.clip(np.round(w / scale + zp), q_min, q_max_val).astype(np.int32)
            w_q = (scale * (q - zp)).astype(np.float32)
            Q_block[:, j] = w_q
            E_block[:, j] = (w - w_q) / C[col, col]
            W_block[:, j+1:] -= np.outer(E_block[:, j], C[col, col+1:col+1+(j_end-i-j-1)])

        Q[:, i:j_end] = Q_block
        W[:, j_end:] -= E_block @ C[i:j_end, j_end:]

    return Q

range_methods = [range_minmax, range_percentile, range_mse]
quantization_methods = [uniform_affine, uniform_symmetric, power_of_two, gptq]