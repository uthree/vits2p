import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from .models import SynthesizerTrn
from .discriminator import Discriminator
from .duration_discriminator import DurationDiscriminator
from module.vits.helper.slice import slice_waveform
from .loss import kl_loss, kl_loss_normal, generator_adversarial_loss, discriminator_adversarial_loss, feature_matching_loss, multiscale_stft_loss, f0_estimation_loss, duration_discriminator_adversarial_loss, duration_generator_adversarial_loss


import lightning as L


class VITS(L.LightningModule):
    def __init__(self, config):
        super().__init__()
        self.frame_size = config.generator.frame_size

        self.net_g = SynthesizerTrn(**config.generator)
        self.net_d = Discriminator(**config.discriminator)
        self.net_dd = DurationDiscriminator(**config.duration_discriminator)

        self.automatic_optimization = False
        self.save_hyperparameters()

    def training_step(self, batch):
        y, spec, spec_length, f0, text, text_length, sid, lang_id = batch
        opt_g, opt_d, opt_dd = self.optimizers()

        (
            y_hat,
            (_, f0_logits, f0_label),
            (loss_dp, loss_sdp, h_text, g, logw, logw_hat),
            attn,
            slice_range,
            x_mask,
            z_mask,
            (m_p_text, logs_p_text),
            (m_p_dur, logs_p_dur, z_q_dur, logs_q_dur),
            (m_p_audio, logs_p_audio, m_q_audio, logs_q_audio),
        ) = self.net_g.forward(text, text_length, spec, spec_length, f0, sid)
        y_hat = y_hat.squeeze(1)
        y_slice = slice_waveform(y, slice_range, self.frame_size)

        loss_stft = multiscale_stft_loss(y_hat, y_slice)
        logits_fake, fmap_fake, dirs_fake = self.net_d(y_hat.detach())
        logits_real, fmap_real, dirs_real = self.net_d(y_slice)
        
        self.toggle_optimizer(opt_d)
        opt_d.zero_grad()
        loss_d = discriminator_adversarial_loss(logits_real, logits_fake, dirs_real, dirs_fake)
        loss_d.backward()
        nn.utils.clip_grad_value_(self.net_d.parameters(), 1.0)
        opt_d.step()
        self.untoggle_optimizer(opt_d)

        self.toggle_optimizer(opt_dd)
        opt_dd.zero_grad()
        dur_logits_real = self.net_dd(h_text, x_mask, logw, g)
        dur_logits_fake = self.net_dd(h_text, x_mask, logw_hat.detach(), g)
        loss_dd = duration_discriminator_adversarial_loss(dur_logits_real, dur_logits_fake)
        loss_dd.backward()
        nn.utils.clip_grad_value_(self.net_dd.parameters(), 1.0)
        opt_dd.step()
        self.untoggle_optimizer(opt_dd)

        logits_fake, fmap_fake, dirs_fake = self.net_d(y_hat)
        loss_adv = generator_adversarial_loss(logits_fake)
        loss_feat = feature_matching_loss(fmap_real, fmap_fake)
        loss_f0 = f0_estimation_loss(f0_logits, f0_label)
        
        loss_kl_dur = kl_loss(z_q_dur, logs_q_dur, m_p_dur, logs_p_dur, z_mask)
        loss_kl_audio = kl_loss_normal(m_p_audio, logs_p_audio, m_q_audio, logs_q_audio, z_mask)
        dur_logits_fake = self.net_dd(h_text, x_mask, logw_hat, g)
        loss_dur_adv = duration_generator_adversarial_loss(dur_logits_fake)

        loss_g = loss_stft * 45 + loss_adv + loss_feat + loss_kl_dur + loss_kl_audio + loss_f0 + loss_dp + loss_sdp + loss_dur_adv

        self.toggle_optimizer(opt_g)
        opt_g.zero_grad()
        loss_g.backward()
        nn.utils.clip_grad_value_(self.net_g.parameters(), 1.0)
        opt_g.step()
        self.untoggle_optimizer(opt_g)

        self.log("generator/stft", loss_stft.item())
        self.log("generator/adversarial", loss_adv.item())
        self.log("generator/feature matching", loss_feat.item())
        self.log("generator/kl_dur", loss_kl_dur.item())
        self.log("generator/kl_audio", loss_kl_audio.item())
        self.log("generator/pitch prediction", loss_f0.item())
        self.log("generator/stochastic duration predictor", loss_sdp.item())
        self.log("generator/duration predictor", loss_dp.item())
        self.log("discriminator/waveform", loss_d.item())
        self.log("discriminator/duration", loss_dd.item())

    def configure_optimizers(self):
        lr = 1e-4
        betas = (0.8, 0.99)

        opt_g = optim.AdamW(self.net_g.parameters(), lr, betas)
        opt_d = optim.AdamW(self.net_d.parameters(), lr, betas)
        opt_dd = optim.AdamW(self.net_dd.parameters(), lr, betas)
        
        return opt_g, opt_d, opt_dd