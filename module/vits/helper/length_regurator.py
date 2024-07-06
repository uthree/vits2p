import math
import torch
import torch.nn.functional as F
import numpy as np

from module.monotonic_align import maximum_path
from .model import sequence_mask, convert_pad_shape


# * Ready and Tested
def search_path(z_p, m_p, logs_p, x_mask, y_mask, mas_noise_scale=0.01):
    with torch.no_grad():
        o_scale = torch.exp(-2 * logs_p)  # [b, d, t]
        logp1 = torch.sum(-0.5 * math.log(2 * math.pi) - logs_p, [1], keepdim=True)  # [b, 1, t]
        logp2 = torch.matmul(-0.5 * (z_p**2).mT, o_scale)  # [b, t', d] x [b, d, t] = [b, t', t]
        logp3 = torch.matmul(z_p.mT, (m_p * o_scale))  # [b, t', d] x [b, d, t] = [b, t', t]
        logp4 = torch.sum(-0.5 * (m_p**2) * o_scale, [1], keepdim=True)  # [b, 1, t]
        logp = logp1 + logp2 + logp3 + logp4  # [b, t', t]

        if mas_noise_scale > 0.0:
            epsilon = torch.std(logp) * torch.randn_like(logp) * mas_noise_scale
            logp = logp + epsilon

        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)  # [b, 1, t] * [b, t', 1] = [b, t', t]
        attn = maximum_path(logp, attn_mask.squeeze(1)).unsqueeze(1).detach()  # [b, 1, t', t] maximum_path_cuda
    return attn


def generate_path(duration: torch.Tensor, mask: torch.Tensor):
    """
    duration: [b, 1, t_x]
    mask: [b, 1, t_y, t_x]
    """
    b, _, t_y, t_x = mask.shape
    cum_duration = torch.cumsum(duration, -1)

    cum_duration_flat = cum_duration.view(b * t_x)
    path = sequence_mask(cum_duration_flat, t_y).to(mask.dtype)
    path = path.view(b, t_x, t_y)
    path = path - F.pad(path, convert_pad_shape([[0, 0], [1, 0], [0, 0]]))[:, :-1]
    path = path.unsqueeze(1).mT * mask
    return path