import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import List


def safe_log(x, eps=1e-6):
    return torch.log(x + eps)


def multiscale_stft_loss(x: torch.Tensor, y: torch.Tensor, scales=[128, 256, 512], alpha=1.0, beta=1.0):
    '''
    shapes:
        x: [N, Waveform Length]
        y: [N, Waveform Length]

        Output: []
    '''
    x = x.to(torch.float)
    y = y.to(torch.float)

    loss = 0
    num_scales = len(scales)
    for s in scales:
        hop_length = s
        n_fft = s * 4
        window = torch.hann_window(n_fft, device=x.device)
        x_spec = torch.stft(x, n_fft, hop_length, return_complex=True, window=window).abs()
        y_spec = torch.stft(y, n_fft, hop_length, return_complex=True, window=window).abs()

        x_spec[x_spec.isnan()] = 0
        x_spec[x_spec.isinf()] = 0
        y_spec[y_spec.isnan()] = 0
        y_spec[y_spec.isinf()] = 0

        loss += F.l1_loss(safe_log(x_spec), safe_log(y_spec)) * alpha + F.mse_loss(x_spec, y_spec) * beta 
    return loss / num_scales


# SAN Loss + LS-GAN Loss
def discriminator_adversarial_loss(real_logits, fake_logits, real_dirs, fake_dirs):
    loss = 0.0
    for lr, lf, dr, df,  in zip(real_logits, fake_logits, real_dirs, fake_dirs):
        real_loss = ((lr - 1.0) ** 2).mean() - dr.mean()
        fake_loss = ((lf + 1.0) ** 2).mean() + df.mean()
        loss += F.relu(real_loss) + F.relu(fake_loss)
    return loss


# LS-GAN Loss
def generator_adversarial_loss(fake_logits):
    loss = 0.0
    for dg in fake_logits:
        fake_loss = ((dg - 1.0) ** 2).mean()
        loss += F.relu(fake_loss)
    return loss

    
def feature_matching_loss(fmap_real, fmap_fake):
    loss = 0
    for r, f in zip(fmap_real, fmap_fake):
        f = f.float()
        r = r.float()
        loss += F.l1_loss(f, r)
    return loss


def kl_loss(z_p: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor, z_mask: torch.Tensor):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    z_p = z_p.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * ((z_p - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def kl_loss_normal(m_q: torch.Tensor, logs_q: torch.Tensor, m_p: torch.Tensor, logs_p: torch.Tensor, z_mask: torch.Tensor):
    """
    z_p, logs_q: [b, h, t_t]
    m_p, logs_p: [b, h, t_t]
    """
    m_q = m_q.float()
    logs_q = logs_q.float()
    m_p = m_p.float()
    logs_p = logs_p.float()
    z_mask = z_mask.float()

    kl = logs_p - logs_q - 0.5
    kl += 0.5 * (torch.exp(2.0 * logs_q) + (m_q - m_p) ** 2) * torch.exp(-2.0 * logs_p)
    kl = torch.sum(kl * z_mask)
    l = kl / torch.sum(z_mask)
    return l


def f0_estimation_loss(f0_logits, f0_label):
    w = torch.ones(f0_logits.shape[1], device=f0_logits.device)
    w[0] = 0.05
    return F.cross_entropy(f0_logits, f0_label, w)


def duration_discriminator_adversarial_loss(logits_real, logits_fake):
    loss = (logits_real ** 2).mean() + ((logits_fake - 1) ** 2).mean()
    return loss / 2.0

def duration_generator_adversarial_loss(logits_fake):
    return (logits_fake ** 2).mean()