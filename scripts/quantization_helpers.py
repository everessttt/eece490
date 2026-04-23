import torch
from PoTPTQ import PoTPTQ

# https://arxiv.org/pdf/2106.08295
# https://arxiv.org/html/2507.11959v1

import torch

def uniform_affine(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    t_min = tensor.min()
    t_max = tensor.max()
    if t_max == t_min:
        return torch.full_like(tensor, t_min.item())
    q_min = 0
    q_max = (1 << num_bits) - 1
    scale = torch.clamp((t_max - t_min) / (q_max - q_min), min=1e-8)
    zp = torch.round(-t_min / scale).clamp(q_min, q_max).to(torch.int64)
    q = torch.clamp(torch.round(tensor / scale + zp), q_min, q_max)
    return (scale * (q - zp)).to(torch.float32)

def uniform_symmetric(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    abs_max = float(tensor.abs().max())
    if abs_max == 0:
        return torch.zeros_like(tensor)
    q_max = (1 << (num_bits - 1)) - 1
    q_min = -(1 << (num_bits - 1))
    scale = abs_max / q_max
    q = torch.clamp(torch.round(tensor / scale), q_min, q_max).to(torch.int64)
    return (q * scale).to(torch.float32)

def power_of_two(tensor: torch.Tensor, num_bits: int) -> torch.Tensor:
    abs_max = float(tensor.abs().max())
    if abs_max == 0:
        return torch.zeros_like(tensor)
    q_max = (1 << (num_bits - 1)) - 1
    q_min = -(1 << (num_bits - 1))
    exp = torch.ceil(torch.log2(torch.tensor(abs_max / q_max))).item()
    scale = 2.0 ** exp
    q = torch.clamp(torch.round(tensor / scale), q_min, q_max).to(torch.int64)
    return (q * scale).to(torch.float32)

quantization_methods = [uniform_affine, uniform_symmetric, power_of_two, PoTPTQ]