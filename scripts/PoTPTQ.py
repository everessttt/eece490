import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Adam
import math
from pathlib import Path

@torch.no_grad()
def PoTPTQ_1(tensor: torch.Tensor, num_bits: int, group_size: int = 128) -> torch.Tensor:

    if tensor.abs().max() == 0:
        return torch.zeros_like(tensor)

    if num_bits > 3:
        q_max = (1 << (num_bits - 1)) - 1
        q_min = -(1 << (num_bits - 1))
        abs_max = tensor.abs().max()
        scale = 2.0 ** math.ceil(math.log2((abs_max / q_max).item() + 1e-12))
        q = torch.round(tensor / scale).clamp(q_min, q_max)
        return q.to(torch.float32) * scale

    # Per-channel quantization for multi-dimensional tensors
    if tensor.ndim > 1:
        out = torch.zeros_like(tensor)
        for i in range(tensor.shape[0]):
            out[i] = PoTPTQ_1(tensor[i], num_bits, group_size)
        return out

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
    S0 = (abs_max / (2.0 ** (q_max - 1))).clamp(min=1e-8)

    B = torch.arange(1, 201, device=flat.device, dtype=torch.float32) * 0.01

    sb = S0[:, None] * B[None, :]
    abs_exp = abs_g[:, None, :]
    sb_exp = sb[:, :, None]

    eps = 1e-12
    ratio = abs_exp / (sb_exp + eps)
    log_vals = torch.log2(ratio.clamp(min=eps))
    log_vals = torch.where(abs_exp > 0, log_vals, torch.zeros_like(log_vals))
    E = torch.clamp(torch.round(log_vals), 0, q_max)

    w_sign = torch.sign(groups)[:, None, :]
    w_hat = sb_exp * w_sign * (2.0 ** E)

    errs = ((groups[:, None, :] - w_hat) ** 2).sum(dim=2)
    best_idx = errs.argmin(dim=1)

    out = w_hat[torch.arange(G, device=flat.device), best_idx]
    return out.reshape(-1)[:n].reshape(orig_shape).to(torch.float32)

def PoTPTQ_2(
    model: nn.Module,
    calibration_imgs: torch.Tensor,
    num_bits: int,
    group_size: int = 128,
    lr: float = 5e-3,
    lam: float = 1e-2,
    epochs: int = 10,
    ref_outputs: dict | None = None,
) -> nn.Module:

    device = torch.device("cpu")
    model = model.to(device)
    calibration_imgs = calibration_imgs.to(device)

    if ref_outputs is not None:
        ref_outputs = {
            k: (x.to(device), y.to(device))
            for k, (x, y) in ref_outputs.items()
        }

    model.eval()
    eps = 1e-12
    q_max = (1 << (num_bits - 1)) - 1

    for name, module in model.named_modules():
        if not isinstance(module, (nn.Conv2d, nn.Linear)):
            continue

        W = module.weight.data.clone()
        if W.ndim < 2:
            continue

        print(f"\n[PoTPTQ] refining {name} {tuple(W.shape)}")

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
        S = (S0 * B[best_idx]).detach()

        # Use pre-quantization reference if provided
        if ref_outputs is not None and name in ref_outputs:
            X_in, H_orig = ref_outputs[name]
            print(f"  using ref_outputs for {name}")
        else:
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
            print(f"  captured activations for {name}")

        # Sanity check
        with torch.no_grad():
            flat_w = W.reshape(-1)
            if pad:
                flat_w = torch.cat([flat_w, torch.zeros(pad, device=W.device)])
            groups_w = flat_w.view(G, group_size)
            abs_w = groups_w.abs()
            log_w = torch.log2((abs_w / (S[:, None].expand(G, group_size) + eps)).clamp(min=eps))
            log_w = torch.where(abs_w > 0, log_w, torch.zeros_like(log_w))
            E_check = torch.clamp(torch.round(log_w), 0, q_max)
            W_check = S[:, None].expand(G, group_size) * torch.sign(groups).detach() * (2.0 ** E_check)
            W_check = W_check.reshape(-1)[:n].reshape(W.shape)
            if isinstance(module, nn.Conv2d):
                H_check = F.conv2d(X_in, W_check, bias=module.bias,
                                   stride=module.stride, padding=module.padding,
                                   dilation=module.dilation, groups=module.groups)
            else:
                H_check = F.linear(X_in, W_check, module.bias)
            init_loss = ((H_orig - H_check) ** 2).mean().item()
            print(f"  init loss (>0 means ref is different from quantized): {init_loss:.6f}")

        Gamma = nn.Parameter(torch.zeros(G, device=W.device))
        opt = Adam([Gamma], lr=lr, weight_decay=0)
        sign_g = torch.sign(groups).detach()
        layer_weight = 10.0 if "23" in name else 1.0

        for epoch in range(epochs):
            opt.zero_grad()

            S_hat = S * (1 + Gamma)
            S_hat_exp = S_hat[:, None].expand(G, group_size)

            flat_w = W.reshape(-1)
            if pad:
                flat_w = torch.cat([flat_w, torch.zeros(pad, device=W.device)])
            groups_w = flat_w.view(G, group_size)
            abs_w = groups_w.abs()

            with torch.no_grad():
                log_w = torch.log2((abs_w / (S_hat_exp.detach() + eps)).clamp(min=eps))
                log_w = torch.where(abs_w > 0, log_w, torch.zeros_like(log_w))
                E_fixed = torch.clamp(torch.round(log_w), 0, q_max)

            W_hat_groups = S_hat_exp * sign_g * (2.0 ** E_fixed)
            W_hat = W_hat_groups.reshape(-1)[:n].reshape(W.shape)

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

            loss = layer_weight * ((H_orig - H_q) ** 2).mean() + (lam / 2) * (Gamma ** 2).sum()
            loss.backward()
            opt.step()

            print(f"  epoch {epoch + 1}/{epochs} loss={loss.item():.6f} "
                  f"gamma_mean={Gamma.data.mean().item():.6f} "
                  f"gamma_grad={Gamma.grad.abs().mean().item() if Gamma.grad is not None else 0:.6f}",
                  flush=True)

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

    model = model.cpu()
    return model

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

def PoTPTQ(tensor: torch.Tensor, num_bits: int, group_size: int = 128) -> torch.Tensor:
    """Step 1 only — Step 2 applied model-wise in main.py via PoTPTQ_2."""
    return PoTPTQ_1(tensor, num_bits, group_size)