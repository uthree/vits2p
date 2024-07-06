import torch
import torch.nn as nn
import torch.nn.functional as F

from .normalization import LayerNorm
from .helper.model import get_padding


class ResBlock(nn.Module):
    def __init__(self, channels, kernel_size=7):
        super().__init__()
        self.norm = LayerNorm(channels)
        self.c1 = nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, 1), groups=channels)
        self.c2 = nn.Conv1d(channels, channels, 1)
        self.c3 = nn.Conv1d(channels, channels, 1)

    def forward(self, x):
        res = x
        x = self.c1(x)
        x = self.norm(x)
        x = self.c2(x)
        x = F.leaky_relu(x, 0.1)
        x = self.c3(x)
        return x + res


class PitchPredictor(nn.Module):
    def __init__(
            self,
            inter_channels=192,
            gin_channels=192,
            num_layers=4,
            num_classes=512,
            classes_per_octave=48,
            min_frequency=20.0
        ):
        super().__init__()
        self.num_classes = num_classes
        self.classes_per_octave = classes_per_octave
        self.min_frequency = min_frequency
        
        self.g_in = nn.Conv1d(gin_channels, inter_channels, 1)
        self.layers = nn.ModuleList()
        for _ in range(num_layers):
            self.layers.append(ResBlock(inter_channels))
        self.proj = nn.Conv1d(inter_channels, num_classes, 1)

    # spec: [BatchSize, fft_bin, Length]
    def forward(self, x, g=None):
        x = x
        if g is not None:
            x = x + self.g_in(g)
        for layer in self.layers:
            x = layer(x)
        f0_logits = self.proj(x)
        f0 = self.decode(f0_logits)
        return f0, f0_logits

    # f: [<Any shape allowed>]
    def freq2id(self, f):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        nc = self.num_classes
        return torch.ceil(torch.clamp(cpo * torch.log2(f / fmin), 0, nc-1)).to(torch.long)
    
    # ids: [<Any shape allowed>]
    def id2freq(self, ids):
        fmin = self.min_frequency
        cpo = self.classes_per_octave
        x = ids.to(torch.float)
        x = fmin * (2 ** (x / cpo))
        x[x <= self.min_frequency] = 0
        return x
    
    # z_p: [BatchSize, content_channels, Length]
    # Outputs:
    #   f0: [BatchSize, 1, Length]
    #   energy: [BatchSize, 1, Length]
    def decode(self, logits, k=4):
        probs, indices = torch.topk(logits, k, dim=1)
        probs = F.softmax(probs, dim=1)
        freqs = self.id2freq(indices)
        f0 = (probs * freqs).sum(dim=1, keepdim=True)
        f0[f0 <= self.min_frequency] = 0
        return f0
    
    def infer(self, spec):
        logits = self.forward(spec)
        f0 = self.decode(logits)
        return f0