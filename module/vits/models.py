import torch
from torch import nn

from .encoders import TextEncoder, PosteriorEncoder, AudioEncoder
from .normalizing_flows import ResidualCouplingBlock
from .duration_predictors import DurationPredictor, StochasticDurationPredictor
from .pitch_predictor import PitchPredictor
from .decoder import Generator
from .helper.length_regurator import search_path, generate_path
from .helper.model import sequence_mask
from .helper.slice import decide_slice_range, slice_features



class SynthesizerTrn(nn.Module):
    """
    Synthesizer for Training
    """

    def __init__(
        self,
        n_vocab,
        spec_channels,
        segment_size,
        inter_channels,
        hidden_channels,
        filter_channels,
        n_heads,
        n_layers,
        n_layers_q,
        n_flows,
        kernel_size,
        p_dropout,
        speaker_cond_layer,
        resblock,
        resblock_kernel_sizes,
        resblock_dilation_sizes,
        sample_rate,
        frame_size,
        upsample_rates,
        upsample_initial_channel,
        upsample_kernel_sizes,
        mas_noise_scale,
        mas_noise_scale_decay,
        use_transformer_flow=True,
        n_speakers=0,
        gin_channels=0,
        **kwargs
    ):
        super().__init__()
        self.frame_size = frame_size
        self.sample_rate = sample_rate

        self.segment_size = segment_size
        self.n_speakers = n_speakers
        self.mas_noise_scale = mas_noise_scale
        self.mas_noise_scale_decay = mas_noise_scale_decay

        self.enc_p = TextEncoder(n_vocab, inter_channels, hidden_channels, filter_channels, n_heads, n_layers, kernel_size, p_dropout, gin_channels=gin_channels, speaker_cond_layer=speaker_cond_layer)
        self.enc_q = PosteriorEncoder(spec_channels, inter_channels, hidden_channels, 5, 1, n_layers_q, gin_channels=gin_channels)
        # self.enc_q = AudioEncoder(spec_channels, inter_channels, 32, 768, n_heads, 2, kernel_size, p_dropout, gin_channels=gin_channels)
        # self.enc_q = AudioEncoder(spec_channels, inter_channels, 32, 32, n_heads, 3, kernel_size, p_dropout, gin_channels=gin_channels)
        self.dec = Generator(inter_channels, gin_channels, upsample_initial_channel, sample_rate, resblock, resblock_kernel_sizes, resblock_dilation_sizes, upsample_kernel_sizes, upsample_rates)
        self.flow = ResidualCouplingBlock(inter_channels, hidden_channels, 5, 1, 4, n_flows=n_flows, gin_channels=gin_channels, mean_only=False, use_transformer_flow=use_transformer_flow)
        
        # pitch predictor
        self.pitch_predictor = PitchPredictor(inter_channels, gin_channels)

        self.sdp = StochasticDurationPredictor(hidden_channels, hidden_channels, 3, 0.5, 4, gin_channels=gin_channels)
        self.dp = DurationPredictor(hidden_channels, 256, 3, 0.5, gin_channels=gin_channels)

        if n_speakers > 1:
            self.emb_g = nn.Embedding(n_speakers, gin_channels)

    def forward(self, x, x_lengths, y, y_lengths, f0, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z_p_text, m_p_text, logs_p_text, h_text, x_mask = self.enc_p(x, x_lengths, g=g)
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.enc_q(y, y_lengths, g=g)
        z_q_dur, m_q_dur, logs_q_dur = self.flow(z_q_audio, m_q_audio, logs_q_audio, y_mask, g=g)

        attn = search_path(z_q_dur, m_p_text, logs_p_text, x_mask, y_mask, mas_noise_scale=self.mas_noise_scale)
        self.mas_noise_scale = max(self.mas_noise_scale - self.mas_noise_scale_decay, 0.0)

        w = attn.sum(2)  # [b, 1, t_s]

        # * reduce posterior
        # TODO Test gain constant
        if False:
            attn_inv = attn.squeeze(1) * (1 / (w + 1e-9))
            m_q_text = torch.matmul(attn_inv.mT, m_q_dur.mT).mT
            logs_q_text = torch.matmul(attn_inv.mT, logs_q_dur.mT).mT

        # * expand prior
        l_length_sdp = self.sdp(h_text, x_mask, w, g=g)
        l_length_sdp = l_length_sdp / torch.sum(x_mask)

        logw = torch.log(w + 1e-6) * x_mask
        logw_hat = self.dp(h_text, x_mask, g=g)
        l_length_dp = torch.sum((logw - logw_hat) ** 2, [1, 2]) / torch.sum(x_mask)  # for averaging

        m_p_dur = torch.matmul(attn.squeeze(1), m_p_text.mT).mT
        logs_p_dur = torch.matmul(attn.squeeze(1), logs_p_text.mT).mT
        z_p_dur = m_p_dur + torch.randn_like(m_p_dur) * torch.exp(logs_p_dur) * y_mask

        z_p_audio, m_p_audio, logs_p_audio = self.flow(z_p_dur, m_p_dur, logs_p_dur, y_mask, g=g, reverse=True)
        _, f0_logits = self.pitch_predictor(z_p_dur, g=g)
        f0_label = self.pitch_predictor.freq2id(f0).squeeze(1)

        slice_range = decide_slice_range(z_q_audio.shape[2], self.segment_size)
        z_slice = slice_features(z_q_audio, slice_range)
        f0_slice = slice_features(f0, slice_range)

        o = self.dec(z_slice, f0_slice, g=g)

        l_length_dp = l_length_dp.mean()
        l_length_sdp = l_length_sdp.mean()
        return (
            o,
            (f0, f0_logits, f0_label),
            (l_length_dp, l_length_sdp, h_text, g, logw, logw_hat),
            attn,
            slice_range,
            x_mask,
            y_mask,
            (m_p_text, logs_p_text),
            (m_p_dur, logs_p_dur, z_q_dur, logs_q_dur),
            (m_p_audio, logs_p_audio, m_q_audio, logs_q_audio),
        )

    def infer(self, x, x_lengths, sid=None, noise_scale=1, length_scale=1, use_sdp=False, noise_scale_w=1.0, max_len=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None

        z_p_text, m_p_text, logs_p_text, h_text, x_mask = self.enc_p(x, x_lengths, g=g)

        if use_sdp:
            logw = self.sdp(h_text, x_mask, g=g, reverse=True, noise_scale=noise_scale_w)
        else:
            logw = self.dp(h_text, x_mask, g=g)
        w = torch.exp(logw) * x_mask * length_scale
        w_ceil = torch.ceil(w)
        y_lengths = torch.clamp_min(torch.sum(w_ceil, [1, 2]), 1).long()
        y_mask = torch.unsqueeze(sequence_mask(y_lengths, None), 1).to(x_mask.dtype)
        attn_mask = torch.unsqueeze(x_mask, 2) * torch.unsqueeze(y_mask, -1)
        attn = generate_path(w_ceil, attn_mask)

        m_p_dur = torch.matmul(attn.squeeze(1), m_p_text.mT).mT  # [b, t', t], [b, t, d] -> [b, d, t']
        logs_p_dur = torch.matmul(attn.squeeze(1), logs_p_text.mT).mT  # [b, t', t], [b, t, d] -> [b, d, t']
        z_p_dur = m_p_dur + torch.randn_like(m_p_dur) * torch.exp(logs_p_dur) * noise_scale

        z_p_audio, m_p_audio, logs_p_audio = self.flow(z_p_dur, m_p_dur, logs_p_dur, y_mask, g=g, reverse=True)
        f0, _ = self.pitch_predictor(z_p_dur, g=g)
        o = self.dec((z_p_audio * y_mask)[:, :, :max_len], f0, g=g)
        return o, attn, y_mask, (z_p_dur, m_p_dur, logs_p_dur), (z_p_audio, m_p_audio, logs_p_audio)

    def voice_conversion(self, y, y_lengths, f0, sid_src, sid_tgt):
        assert self.n_speakers > 0, "n_speakers have to be larger than 0."
        g_src = self.emb_g(sid_src).unsqueeze(-1)
        g_tgt = self.emb_g(sid_tgt).unsqueeze(-1)
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.enc_q(y, y_lengths, g=g_src)
        z_q_dur, m_q_dur, logs_q_dur = self.flow(z_q_audio, m_q_audio, logs_q_audio, y_mask, g=g_src)
        z_p_audio, m_p_audio, logs_p_audio = self.flow(z_q_dur, m_q_dur, logs_q_dur, y_mask, g=g_tgt, reverse=True)
        o_hat = self.dec(z_p_audio * y_mask, f0, g=g_tgt)
        return o_hat, y_mask, (z_q_dur, m_q_dur, logs_q_dur), (z_p_audio, m_p_audio, logs_p_audio)

    def voice_restoration(self, y, y_lengths, f0, sid=None):
        if self.n_speakers > 0:
            g = self.emb_g(sid).unsqueeze(-1)  # [b, h, 1]
        else:
            g = None
        z_q_audio, m_q_audio, logs_q_audio, y_mask = self.enc_q(y, y_lengths, g=g)
        z_q_dur, m_q_dur, logs_q_dur = self.flow(z_q_audio, m_q_audio, logs_q_audio, y_mask, g=g)
        z_p_audio, m_p_audio, logs_p_audio = self.flow(z_q_dur, m_q_dur, logs_q_dur, y_mask, g=g, reverse=True)
        o_hat = self.dec(z_p_audio * y_mask, f0, g=g)
        return o_hat, y_mask, (z_q_dur, m_q_dur, logs_q_dur), (z_p_audio, m_p_audio, logs_p_audio)
