from torch import nn
import torch
from torch.nn.utils.parametrizations import weight_norm, spectral_norm
from torch.nn import Conv1d, AvgPool1d, Conv2d
import torch.nn.functional as F

from src.models.modules import WNConv1d
from src.models.utils import weights_init
from src.utils import capture_init


# Melgan discriminator

class NLayerDiscriminator(nn.Module):
    def __init__(self, ndf, n_layers, downsampling_factor, channels=1):
        super().__init__()
        model = nn.ModuleDict()
        model["layer_0"] = nn.Sequential(
            nn.ReflectionPad1d(7),
            WNConv1d(channels, ndf, kernel_size=15),
            nn.LeakyReLU(0.2, True),
        )

        nf = ndf
        stride = downsampling_factor
        max_nf = (stride ** (n_layers -1) ) *ndf
        for n in range(1, n_layers + 1):
            nf_prev = nf
            nf = min(nf * stride, max_nf)
            model["layer_%d" % n] = nn.Sequential(
                WNConv1d(
                    nf_prev,
                    nf,
                    kernel_size=stride * 10 + 1,
                    stride=stride,
                    padding=stride * 5,
                    groups=nf_prev // 4,
                ),
                nn.LeakyReLU(0.2, True),
            )
        nf = min(nf * 2, max_nf)
        model["layer_%d" % (n_layers + 1)] = nn.Sequential(
            WNConv1d(nf_prev, nf, kernel_size=5, stride=1, padding=2),
            nn.LeakyReLU(0.2, True),
        )
        model["layer_%d" % (n_layers + 2)] = WNConv1d(
            nf, 1, kernel_size=3, stride=1, padding=1
        )
        self.model = model

    def forward(self, x):
        results = []
        for key, layer in self.model.items():
            x = layer(x)
            results.append(x)
        return results


class Discriminator(nn.Module):
    @capture_init
    def __init__(self, num_D, ndf, n_layers, downsampling_factor, channels=1):
        super().__init__()
        self.model = nn.ModuleDict()
        self.num_D = num_D
        for i in range(num_D):
            self.model[f"disc_{i}"] = NLayerDiscriminator(
                ndf, n_layers, downsampling_factor, channels=channels
            )

        self.downsample = AvgPool1d(4, stride=2, padding=1, count_include_pad=False)
        self.apply(weights_init)

    def forward(self, x):
        results = []
        for key, disc in self.model.items():
            results.append(disc(x))
            x = self.downsample(x)
        return results

# HiFiGAN discriminators

LRELU_SLOPE = 0.1


def get_padding(kernel_size, dilation=1):
    return int((kernel_size * dilation - dilation) / 2)


class DiscriminatorP(torch.nn.Module):
    @capture_init
    def __init__(self, period, kernel_size=5, stride=3, use_spectral_norm=False, hidden=32, channels=1):
        super(DiscriminatorP, self).__init__()
        self.period = period
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm
        self.convs = nn.ModuleList([
            norm_f(Conv2d(channels, hidden, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(hidden, hidden * 4, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(hidden * 4, hidden * 16, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(hidden * 16, hidden * 32, (kernel_size, 1), (stride, 1), padding=(get_padding(5, 1), 0))),
            norm_f(Conv2d(hidden * 32, hidden * 32, (kernel_size, 1), 1, padding=(2, 0))),
        ])
        self.conv_post = norm_f(Conv2d(hidden * 32, 1, (3, 1), 1, padding=(1, 0)))

    def forward(self, x):
        fmap = []

        # 1d to 2d
        b, c, t = x.shape
        if t % self.period != 0:  # pad first
            n_pad = self.period - (t % self.period)
            x = F.pad(x, (0, n_pad), "reflect")
            t = t + n_pad
        x = x.view(b, c, t // self.period, self.period)

        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiPeriodDiscriminator(torch.nn.Module):
    @capture_init
    def __init__(self, hidden=32, periods=[2, 3, 5, 7, 11], channels=1):
        super(MultiPeriodDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorP(period, hidden=hidden, channels=channels) for period in periods
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


class DiscriminatorS(torch.nn.Module):
    @capture_init
    def __init__(self, use_spectral_norm=False, hidden=128, channels=1):
        super(DiscriminatorS, self).__init__()
        norm_f = weight_norm if use_spectral_norm == False else spectral_norm

        self.convs = nn.ModuleList([
            norm_f(Conv1d(channels, hidden, 15, 1, padding=7)),
            norm_f(Conv1d(hidden, hidden, 41, 2, groups=4, padding=20)),
            norm_f(Conv1d(hidden, hidden * 2, 41, 2, groups=16, padding=20)),
            norm_f(Conv1d(hidden * 2, hidden * 4, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(hidden * 4, hidden * 8, 41, 4, groups=16, padding=20)),
            norm_f(Conv1d(hidden * 8, hidden * 8, 41, 1, groups=16, padding=20)),
            norm_f(Conv1d(hidden * 8, hidden * 8, 5, 1, padding=2)),
        ])
        self.conv_post = norm_f(Conv1d(hidden * 8, 1, 3, 1, padding=1))

    def forward(self, x):
        fmap = []
        for l in self.convs:
            x = l(x)
            x = F.leaky_relu(x, LRELU_SLOPE)
            fmap.append(x)
        x = self.conv_post(x)
        fmap.append(x)
        x = torch.flatten(x, 1, -1)

        return x, fmap


class MultiScaleDiscriminator(torch.nn.Module):
    @capture_init
    def __init__(self, hidden=64, num_D=3, channels=1):
        super(MultiScaleDiscriminator, self).__init__()
        self.discriminators = nn.ModuleList([
            DiscriminatorS(use_spectral_norm=i == 0, hidden=hidden, channels=channels) for i in range(num_D)
        ])
        self.meanpools = nn.ModuleList([
            AvgPool1d(4, 2, padding=2),
            AvgPool1d(4, 2, padding=2)
        ])

    def forward(self, y, y_hat):
        y_d_rs = []
        y_d_gs = []
        fmap_rs = []
        fmap_gs = []
        for i, d in enumerate(self.discriminators):
            if i != 0:
                y = self.meanpools[i - 1](y)
                y_hat = self.meanpools[i - 1](y_hat)
            y_d_r, fmap_r = d(y)
            y_d_g, fmap_g = d(y_hat)
            y_d_rs.append(y_d_r)
            fmap_rs.append(fmap_r)
            y_d_gs.append(y_d_g)
            fmap_gs.append(fmap_g)

        return y_d_rs, y_d_gs, fmap_rs, fmap_gs


def feature_loss(fmap_r, fmap_g):
    loss = 0
    total_n_layers = 0
    for dr, dg in zip(fmap_r, fmap_g):
        for rl, gl in zip(dr, dg):
            total_n_layers += 1
            loss += torch.mean(torch.abs(rl - gl))

    return loss / total_n_layers


def discriminator_loss(disc_real_outputs, disc_generated_outputs):
    loss = 0
    # r_losses = []
    # g_losses = []
    for dr, dg in zip(disc_real_outputs, disc_generated_outputs):
        #r_loss = torch.mean((1 - dr) ** 2)
        #g_loss = torch.mean(dg ** 2)
        #loss += (r_loss + g_loss)
        loss += torch.mean(torch.sigmoid(dr)-torch.sigmoid(dg))
        # r_losses.append(r_loss.item())
        # g_losses.append(g_loss.item())

    return loss  # , r_losses, g_losses


def generator_loss(disc_outputs):
    loss = 0
    # gen_losses = []
    for dg in disc_outputs:
        l = -torch.mean(torch.sigmoid(dg))
        # gen_losses.append(l)
        loss += l

    return loss  # , gen_losses