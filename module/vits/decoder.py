import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import weight_norm
import math
import numpy as np

from module.vits.helper.model import init_weights, get_padding


LRELU_SLOPE = 0.1


class ResBlock1(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3, 5]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        self.convs2 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))
            self.convs2.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, 1), 1)))

    def forward(self, x):
        for c1, c2 in zip(self.convs1, self.convs2):
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            xt = F.leaky_relu(xt, 0.1)
            xt = c2(xt)
            x = x + xt
        return x
    

class ResBlock2(nn.Module):
    def __init__(self, channels, kernel_size=3, dilations=[1, 3]):
        super().__init__()
        self.convs1 = nn.ModuleList()
        for d in dilations:
            self.convs1.append(weight_norm(nn.Conv1d(channels, channels, kernel_size, 1, get_padding(kernel_size, d), d)))

    def forward(self, x):
        for c1 in self.convs1:
            xt = F.leaky_relu(x, 0.1)
            xt = c1(xt)
            x = x + xt
        return x
    

class SinusodialOscillator(nn.Module):
    def __init__(
            self,
            sample_rate=48000,
            frame_size=480,
            min_frequency=20.0,
            voiced_noise_scale=0.03,
            unvoiced_noise_scale=0.3,
        ):
        super().__init__()
        self.sample_rate = sample_rate
        self.frame_size = frame_size
        self.min_frequency = min_frequency
        self.voiced_noise_scale = voiced_noise_scale
        self.unvoiced_noise_scale = unvoiced_noise_scale

    def forward(self, f0):
        '''
            shapes:
                f0: [N, 1, L]
                Output: [N, 1, L * frame_size]
        '''
        f0 = F.relu(f0)
        f0 = F.interpolate(f0, scale_factor=self.frame_size, mode='linear')
        voiced_mask = (f0 > self.min_frequency).to(f0.dtype)
        integrated = torch.cumsum(f0 / self.sample_rate, dim=2)
        rad = 2 * math.pi * (integrated % 1)
        noise = torch.randn_like(rad)
        voiced_part = torch.sin(rad) + noise * self.voiced_noise_scale
        unvoiced_part = noise * self.unvoiced_noise_scale
        source = voiced_part * voiced_mask + unvoiced_part * (1 - voiced_mask)
        return source
    
    
class Generator(nn.Module):
    def __init__(
            self,
            initial_channels=192,
            gin_channels=192,
            upsample_initial_channels=512,
            sample_rate=48000,
            resblock_type="1",
            resblock_kernel_sizes=[3, 7, 11],
            resblock_dilations=[[1, 3, 5], [1, 3, 5], [1, 3, 5]],
            upsample_kernel_sizes=[24, 20, 4, 4],
            upsample_rates=[12, 10, 2, 2],
        ):
        super().__init__()
        self.num_kernels = len(resblock_kernel_sizes)
        self.num_upsamples = len(upsample_rates)

        self.conv_pre = weight_norm(nn.Conv1d(initial_channels, upsample_initial_channels, 7, 1, 3))
        self.g_in = weight_norm(nn.Conv1d(gin_channels, upsample_initial_channels, 1))

        self.frame_size = np.prod(upsample_rates)
        self.oscillator = SinusodialOscillator(sample_rate, self.frame_size)

        if resblock_type == "1":
            resblock = ResBlock1
        elif resblock_type == "2":
            resblock = ResBlock2
        else:
            raise "invalid resblock type"

        self.source_convs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for i, (u, k) in enumerate(zip(upsample_rates, upsample_kernel_sizes)):
            c1 = upsample_initial_channels//(2**i)
            c2 = upsample_initial_channels//(2**(i+1))
            p = (k-u)//2
            if i == len(upsample_rates) - 1:
                self.source_convs.append(weight_norm(nn.Conv1d(1, c2, 7, 1, 3)))
            else:
                up_prod = int(np.prod(upsample_rates[i+1:]))
                self.source_convs.append(weight_norm(nn.Conv1d(1, c2, up_prod * 2, up_prod, up_prod // 2)))
            self.ups.append(weight_norm(nn.ConvTranspose1d(c1, c2, k, u, p)))
        
        self.resblocks = nn.ModuleList()
        for i in range(len(self.ups)):
            ch = upsample_initial_channels//(2**(i+1))
            for j, (k, d) in enumerate(zip(resblock_kernel_sizes, resblock_dilations)):
                self.resblocks.append(resblock(ch, k, d))
        self.conv_post = weight_norm(nn.Conv1d(ch, 1, 7, 1, padding=3))

        self.apply(init_weights)

    def forward(self, x, f0, g=None):
        x = self.conv_pre(x)
        if g is not None:
            x = x + self.g_in(g)
        source = self.oscillator(f0)
        for i in range(self.num_upsamples):
            x = self.ups[i](x) + self.source_convs[i](source)
            x = F.leaky_relu(x, 0.1)
            xs = None
            for j in range(self.num_kernels):
                if xs is None:
                    xs = self.resblocks[i*self.num_kernels+j](x)
                else:
                    xs += self.resblocks[i*self.num_kernels+j](x)
            x = xs / self.num_kernels
        x = F.leaky_relu(x, 0.1)
        x = self.conv_post(x)
        x = torch.tanh(x)
        return x