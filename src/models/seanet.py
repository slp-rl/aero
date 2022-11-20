import torch.nn as nn
import math
from src.models.utils import capture_init, weights_init
from src.models.modules import WNConv1d, WNConvTranspose1d
from torchaudio.functional import resample
from torch.nn import functional as F



class ResnetBlock(nn.Module):
    def __init__(self, dim, dilation=1):
        super().__init__()
        self.block = nn.Sequential(
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(dilation),
            WNConv1d(dim, dim, kernel_size=3, dilation=dilation),
            nn.LeakyReLU(0.2),
            WNConv1d(dim, dim, kernel_size=1),
        )
        self.shortcut = WNConv1d(dim, dim, kernel_size=1)

    def forward(self, x):
        return self.shortcut(x) + self.block(x)


class Seanet(nn.Module):

    @capture_init
    def __init__(self,
                 latent_space_size=128,
                 ngf=32, n_residual_layers=3,
                 resample=1,
                 normalize=True,
                 floor=1e-3,
                 ratios=[8, 8, 2, 2],
                 in_channels=1,
                 out_channels=1,
                 lr_sr=16000,
                 hr_sr=16000,
                 upsample=True):
        super().__init__()

        self.resample = resample
        self.normalize = normalize
        self.floor = floor
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.scale_factor = int(self.hr_sr / self.lr_sr)
        self.upsample = upsample

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        self.ratios = ratios
        mult = int(2 ** len(ratios))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(latent_space_size, mult * ngf, kernel_size=7, padding=0),
        ]

        encoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(mult * ngf, latent_space_size, kernel_size=7, padding=0)
        ]

        self.encoder.insert(0, nn.Sequential(*encoder_wrapper_conv_layer))
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        for i, r in enumerate(ratios):
            encoder_block = [
                nn.LeakyReLU(0.2),
                WNConv1d(mult * ngf // 2,
                         mult * ngf,
                         kernel_size=r * 2,
                         stride=r,
                         padding=r // 2 + r % 2,
                         ),
            ]

            decoder_block = [
                nn.LeakyReLU(0.2),
                WNConvTranspose1d(
                    mult * ngf,
                    mult * ngf // 2,
                    kernel_size=r * 2,
                    stride=r,
                    padding=r // 2 + r % 2,
                    output_padding=r % 2,
                ),
            ]

            for j in range(n_residual_layers - 1, -1, -1):
                encoder_block = [ResnetBlock(mult * ngf // 2, dilation=3 ** j)] + encoder_block

            for j in range(n_residual_layers):
                decoder_block += [ResnetBlock(mult * ngf // 2, dilation=3 ** j)]

            mult //= 2

            self.encoder.insert(0, nn.Sequential(*encoder_block))
            self.decoder.append(nn.Sequential(*decoder_block))

        encoder_wrapper_conv_layer = [
            nn.ReflectionPad1d(3),
            WNConv1d(in_channels, ngf, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.encoder.insert(0, nn.Sequential(*encoder_wrapper_conv_layer))

        decoder_wrapper_conv_layer = [
            nn.LeakyReLU(0.2),
            nn.ReflectionPad1d(3),
            WNConv1d(ngf, out_channels, kernel_size=7, padding=0),
            nn.Tanh(),
        ]
        self.decoder.append(nn.Sequential(*decoder_wrapper_conv_layer))

        self.apply(weights_init)

    def estimate_output_length(self, length):
        """
        Return the nearest valid length to use with the model so that
        there is no time steps left over in a convolutions, e.g. for all
        layers, size of the input - kernel_size % stride = 0.

        If the mixture has a valid length, the estimated sources
        will have exactly the same length.
        """
        depth = len(self.ratios)
        for idx in range(depth - 1, -1, -1):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            length = math.ceil((length - kernel_size + 2 * padding) / stride) + 1
            length = max(length, 1)
        for idx in range(depth):
            stride = self.ratios[idx]
            kernel_size = 2 * stride
            padding = stride // 2 + stride % 2
            output_padding = stride % 2
            length = (length - 1) * stride + kernel_size - 2 * padding + output_padding
        return int(length)

    def pad_to_valid_length(self, signal):
        valid_length = self.estimate_output_length(signal.shape[-1])
        padding_len = valid_length - signal.shape[-1]
        signal = F.pad(signal, (0, padding_len))
        return signal, padding_len

    def forward(self, signal):

        target_len = signal.shape[-1]
        if self.upsample:
            target_len *= self.scale_factor
        if self.normalize:
            mono = signal.mean(dim=1, keepdim=True)
            std = mono.std(dim=-1, keepdim=True)
            signal = signal / (self.floor + std)
        else:
            std = 1
        x = signal
        if self.upsample:
            x = resample(x, self.lr_sr, self.hr_sr)

        x, padding_len = self.pad_to_valid_length(x)
        skips = []
        for i, encode in enumerate(self.encoder):
            skips.append(x)
            x = encode(x)
        for j, decode in enumerate(self.decoder):
            x = decode(x)
            skip = skips.pop(-1)
            x = x + skip
        if target_len < x.shape[-1]:
            x = x[..., :target_len]
        return std * x
