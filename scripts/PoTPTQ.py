import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from pathlib import Path


# =========================================================
# STEP 1 — Data-Agnostic Scale Initialization (Algorithm 1)
# =========================================================

@torch.no_grad()
def PoTPTQ_1(tensor: torch.Tensor, num_bits: int, group_size: int = 128) -> torch.Tensor:

    if tensor.abs().max() == 0:
        return torch.zeros_like(tensor)

    # PoTPTQ grid search targets 2-3 bit; use standard PoT for higher bits
    if num_bits > 3:
        q_max = (1 << (num_bits - 1)) - 1
        q_min = -(1 << (num_bits - 1))
        abs_max = tensor.abs().max()
        scale = 2.0 ** math.ceil(math.log2((abs_max / q_max).item() + 1e-12))
        q = torch.round(tensor / scale).clamp(q_min, q_max)
        return q.to(torch.float32) * scale

    q_max = (1 << (num_bits - 1)) - 1
    orig_shape = tensor.shape
    flat = tensor.reshape(-1)
    n = flat.numel()

    pad = (group_size - n % group_size) % group_size
    if pad:
        flat = torch.cat([flat, torch.zeros(pad, device=flat.device, dtype=flat.dtype)])

    groups = flat.view(-1, group_size)
    G = groups.shape[0]

    abs_g = groups.abs()
    abs_max = abs_g.amax(dim=1)

    # Eq. 10: s0 = max|W_group| / 2^(q_max - 1)
    S0 = (abs_max / (2.0 ** (q_max - 1))).clamp(min=1e-8)

    # Algorithm 1 line 3: B = {0.01 * i | i = 1..200}
    B = torch.arange(1, 201, device=flat.device, dtype=torch.float32) * 0.01

    sb = S0[:, None] * B[None, :]          # (G, 200)
    abs_exp = abs_g[:, None, :]            # (G, 1, S)
    sb_exp = sb[:, :, None]               # (G, 200, 1)

    eps = 1e-12

    # Eq. 6: E = clamp(round(log2(|W| / sb)), 0, q_max)
    ratio = abs_exp / (sb_exp + eps)
    log_vals = torch.log2(ratio.clamp(min=eps))
    log_vals = torch.where(abs_exp > 0, log_vals, torch.zeros_like(log_vals))
    E = torch.clamp(torch.round(log_vals), 0, q_max)

    # Eq. 7: W_hat = sb * sign(W) * 2^E
    w_sign = torch.sign(groups)[:, None, :]
    w_hat = sb_exp * w_sign * (2.0 ** E)

    # Eq. 8-9: Q1(b) = ||W - W_hat||^2_2, select argmin
    errs = ((groups[:, None, :] - w_hat) ** 2).sum(dim=2)
    best_idx = errs.argmin(dim=1)

    out = w_hat[torch.arange(G, device=flat.device), best_idx]
    return out.reshape(-1)[:n].reshape(orig_shape).to(torch.float32)


# =========================================================
# STEP 2 — Data-Dependent Scale Refinement (Algorithm 2)
# =========================================================

def PoTPTQ_2(
    model: nn.Module,
    calibration_imgs: torch.Tensor,
    num_bits: int,
    group_size: int = 128,
    lr: float = 1e-3,
    lam: float = 1e-2,
    epochs: int = 10,
) -> nn.Module:

    model.eval()
    eps = 1e-12
    q_max = (1 << (num_bits - 1)) - 1

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        W = module.weight.data.clone()
        if W.ndim < 2:
            continue

        print(f"[PoTPTQ] refining {name}")

        flat = W.reshape(-1)
        n = flat.numel()
        pad = (group_size - n % group_size) % group_size
        if pad:
            flat = torch.cat([flat, torch.zeros(pad, device=W.device, dtype=W.dtype)])

        groups = flat.view(-1, group_size)
        G = groups.shape[0]

        abs_g = groups.abs()
        abs_max = abs_g.amax(dim=1)
        S0 = (abs_max / (2.0 ** (q_max - 1))).clamp(min=1e-8)

        B = torch.arange(1, 201, device=W.device, dtype=torch.float32) * 0.01
        sb = S0[:, None] * B[None, :]
        abs_exp = abs_g[:, None, :]
        sb_exp = sb[:, :, None]

        ratio = abs_exp / (sb_exp + eps)
        log_vals = torch.log2(ratio.clamp(min=eps))
        log_vals = torch.where(abs_exp > 0, log_vals, torch.zeros_like(log_vals))
        E = torch.clamp(torch.round(log_vals), 0, q_max)
        w_hat = sb_exp * torch.sign(groups)[:, None, :] * (2.0 ** E)
        errs = ((groups[:, None, :] - w_hat) ** 2).sum(dim=2)
        best_idx = errs.argmin(dim=1)

        # Optimal Step 1 scales S
        S = (S0 * B[best_idx]).detach()    # (G,)

        # Capture original layer output (Algorithm 2 line 5)
        cache: dict = {}

        def make_hook():
            def hook(mod, inp, out):
                cache["x"] = inp[0].detach()
                cache["y"] = out.detach()
            return hook

        h = module.register_forward_hook(make_hook())
        with torch.no_grad():
            _ = model(calibration_imgs)
        h.remove()

        X_in = cache["x"]
        H_orig = cache["y"]

        # Algorithm 2 line 1: Gamma = 0
        Gamma = nn.Parameter(torch.zeros(G, device=W.device))
        opt = Adam([Gamma], lr=lr, weight_decay=0)
        sign_g = torch.sign(groups).detach()

        for epoch in range(epochs):
            opt.zero_grad()

            # Algorithm 2 line 6: S_hat = S * (1 + Gamma)
            S_hat = S * (1 + Gamma)                              # (G,)
            S_hat_exp = S_hat[:, None].expand(G, group_size)    # (G, S)

            flat_w = W.reshape(-1)
            if pad:
                flat_w = torch.cat([flat_w, torch.zeros(pad, device=W.device)])
            groups_w = flat_w.view(G, group_size)
            abs_w = groups_w.abs()

            # Algorithm 2 lines 7-8 + STE (Eq. 18)
            log_w = torch.log2((abs_w / (S_hat_exp + eps)).clamp(min=eps))
            log_w = torch.where(abs_w > 0, log_w, torch.zeros_like(log_w))
            E_ste = log_w + (torch.round(log_w) - log_w).detach()
            E_clamped = torch.clamp(E_ste, 0, q_max)

            # Algorithm 2 line 9: W_hat = S_hat * P * 2^E
            W_hat_groups = S_hat_exp * sign_g * (2.0 ** E_clamped)
            W_hat = W_hat_groups.reshape(-1)[:n].reshape(W.shape)

            # Algorithm 2 line 10: H_quant = F(W_hat, X)
            if isinstance(module, nn.Conv2d):
                H_q = F.conv2d(
                    X_in, W_hat,
                    bias=module.bias,
                    stride=module.stride,
                    padding=module.padding,
                    dilation=module.dilation,
                    groups=module.groups,
                )
            else:
                H_q = F.linear(X_in, W_hat, module.bias)

            # Algorithm 2 line 11: Q2 = ||H_orig - H_quant||^2_F + lambda/2 * ||Gamma||^2_F
            loss = ((H_orig - H_q) ** 2).mean() + (lam / 2) * (Gamma ** 2).sum()
            loss.backward()
            opt.step()

            if (epoch + 1) % 5 == 0 or epoch == 0:
                print(f"    epoch {epoch + 1}/{epochs} loss={loss.item():.6f}")

        # Final write-back with refined scales
        with torch.no_grad():
            S_hat_final = (S * (1 + Gamma)).detach()
            S_hat_exp_final = S_hat_final[:, None].expand(G, group_size)

            flat_w = W.reshape(-1)
            if pad:
                flat_w = torch.cat([flat_w, torch.zeros(pad, device=W.device)])
            groups_w = flat_w.view(G, group_size)
            abs_w = groups_w.abs()

            log_w = torch.log2((abs_w / (S_hat_exp_final + eps)).clamp(min=eps))
            log_w = torch.where(abs_w > 0, log_w, torch.zeros_like(log_w))
            E_final = torch.clamp(torch.round(log_w), 0, q_max)
            W_final = S_hat_exp_final * sign_g * (2.0 ** E_final)
            module.weight.data = W_final.reshape(-1)[:n].reshape(W.shape)

    return model


# =========================================================
# Calibration image loader
# =========================================================

def get_calibration_images(n: int = 128, imgsz: int = 640) -> torch.Tensor:
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    import config
    from ultralytics.data import YOLODataset
    from ultralytics.data.utils import check_det_dataset
    from torch.utils.data import DataLoader

    data_info = check_det_dataset(config.COCO128_PATH)
    ds = YOLODataset(
        img_path=data_info["train"],
        imgsz=imgsz,
        augment=False,
        data=data_info,
    )
    loader = DataLoader(
        ds,
        batch_size=n,
        shuffle=False,
        num_workers=0,
        collate_fn=YOLODataset.collate_fn,
    )
    batch = next(iter(loader))
    return batch["img"].float() / 255.0


# =========================================================
# Entry point for quantization_helpers.py
# =========================================================

def PoTPTQ(tensor: torch.Tensor, num_bits: int, group_size: int = 128) -> torch.Tensor:
    """Step 1 only — Step 2 applied model-wise in main.py via PoTPTQ_2."""
    return PoTPTQ_1(tensor, num_bits, group_size)