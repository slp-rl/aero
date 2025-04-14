"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F

from src.models.utils import capture_init
from src.models.spec import spectro, ispectro
from src.models.modules import DConv, ScaledEmbedding, FTB

import logging
logger = logging.getLogger(__name__)


def rescale_conv(conv, reference):
    std = conv.weight.std().detach()
    scale = (std / reference) ** 0.5
    conv.weight.data /= scale
    if conv.bias is not None:
        conv.bias.data /= scale


def rescale_module(module, reference):
    for sub in module.modules():
        if isinstance(sub, (nn.Conv1d, nn.ConvTranspose1d)):
            rescale_conv(sub, reference)


class HEncLayer(nn.Module):
    def __init__(self, chin, chout, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, is_first=False, freq_attn=False, freq_dim=None, norm=True, context=0,
                 dconv_kw={}, pad=True,
                 rewrite=True):
        """Encoder layer. This used both by the time and the frequency branch.

        Args:
            chin: number of input channels.
            chout: number of output channels.
            norm_groups: number of groups for group norm.
            empty: used to make a layer with just the first conv. this is used
                before merging the time and freq. branches.
            freq: this is acting on frequencies.
            dconv: insert DConv residual branches.
            norm: use GroupNorm.
            context: context size for the 1x1 conv.
            dconv_kw: list of kwargs for the DConv class.
            pad: pad the input. Padding is done so that the output size is
                always the input size / stride.
            rewrite: add 1x1 conv at the end of the layer.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if stride == 1 and kernel_size % 2 == 0 and kernel_size > 1:
            kernel_size -= 1
        if pad:
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        klass = nn.Conv2d
        self.chin = chin
        self.chout = chout
        self.freq = freq
        self.kernel_size = kernel_size
        self.stride = stride
        self.empty = empty
        self.freq_attn = freq_attn
        self.freq_dim = freq_dim
        self.norm = norm
        self.pad = pad
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
            if pad != 0:
                pad = [pad, 0]
            # klass = nn.Conv2d
        else:
            kernel_size = [1, kernel_size]
            stride = [1, stride]
            if pad != 0:
                pad = [0, pad]

        self.is_first = is_first

        if is_first:
            self.pre_conv = nn.Conv2d(chin, chout, [1, 1])
            chin = chout

        if self.freq_attn:
            self.freq_attn_block = FTB(input_dim=freq_dim, in_channel=chin)

        self.conv = klass(chin, chout, kernel_size, stride, pad)
        if self.empty:
            return
        self.norm1 = norm_fn(chout)
        self.rewrite = None
        if rewrite:
            self.rewrite = klass(chout, 2 * chout, 1 + 2 * context, 1, context)
            self.norm2 = norm_fn(2 * chout)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chout, **dconv_kw)

    def forward(self, x, inject=None):
        """
        `inject` is used to inject the result from the time branch into the frequency branch,
        when both have the same stride.
        """

        if not self.freq:
            le = x.shape[-1]
            if not le % self.stride == 0:
                x = F.pad(x, (0, self.stride - (le % self.stride)))

        if self.is_first:
            x = self.pre_conv(x)

        if self.freq_attn:
            x = self.freq_attn_block(x)

        x = self.conv(x)

        x = F.gelu(self.norm1(x))
        if self.dconv:
            x = self.dconv(x)

        if self.rewrite:
            x = self.norm2(self.rewrite(x))
            x = F.glu(x, dim=1)

        return x


class HDecLayer(nn.Module):
    def __init__(self, chin, chout, last=False, kernel_size=8, stride=4, norm_groups=1, empty=False,
                 freq=True, dconv=True, norm=True, context=1, dconv_kw={}, pad=True,
                 context_freq=True, rewrite=True):
        """
        Same as HEncLayer but for decoder. See `HEncLayer` for documentation.
        """
        super().__init__()
        norm_fn = lambda d: nn.Identity()  # noqa
        if norm:
            norm_fn = lambda d: nn.GroupNorm(norm_groups, d)  # noqa
        if stride == 1 and kernel_size % 2 == 0 and kernel_size > 1:
            kernel_size -= 1
        if pad:
            pad = (kernel_size - stride) // 2
        else:
            pad = 0
        self.pad = pad
        self.last = last
        self.freq = freq
        self.chin = chin
        self.empty = empty
        self.stride = stride
        self.kernel_size = kernel_size
        self.norm = norm
        self.context_freq = context_freq
        klass = nn.Conv2d
        klass_tr = nn.ConvTranspose2d
        if freq:
            kernel_size = [kernel_size, 1]
            stride = [stride, 1]
        else:
            kernel_size = [1, kernel_size]
            stride = [1, stride]
        self.conv_tr = klass_tr(chin, chout, kernel_size, stride)
        self.norm2 = norm_fn(chout)
        if self.empty:
            return
        self.rewrite = None
        if rewrite:
            if context_freq:
                self.rewrite = klass(chin, 2 * chin, 1 + 2 * context, 1, context)
            else:
                self.rewrite = klass(chin, 2 * chin, [1, 1 + 2 * context], 1,
                                     [0, context])
            self.norm1 = norm_fn(2 * chin)

        self.dconv = None
        if dconv:
            self.dconv = DConv(chin, **dconv_kw)

    def forward(self, x, skip, length):
        if self.freq and x.dim() == 3:
            B, C, T = x.shape
            x = x.view(B, self.chin, -1, T)

        if not self.empty:
            x = torch.cat([x, skip], dim=1)

            if self.rewrite:
                y = F.glu(self.norm1(self.rewrite(x)), dim=1)
            else:
                y = x
            if self.dconv:
                y = self.dconv(y)
        else:
            y = x
            assert skip is None
        z = self.norm2(self.conv_tr(y))
        if self.freq:
            if self.pad:
                z = z[..., self.pad:-self.pad, :]
        else:
            z = z[..., self.pad:self.pad + length]
            assert z.shape[-1] == length, (z.shape[-1], length)
        if not self.last:
            z = F.gelu(z)
        return z


class Aero(nn.Module):
    """
    Deep model for Audio Super Resolution.
    """

    @capture_init
    def __init__(self,
                 # Channels
                 in_channels=1,
                 out_channels=1,
                 audio_channels=2,
                 channels=48,
                 growth=2,
                 # STFT
                 nfft=512,
                 hop_length=64,
                 end_iters=0,
                 cac=True,
                 # Main structure
                 rewrite=True,
                 hybrid=False,
                 hybrid_old=False,
                 # Frequency branch
                 freq_emb=0.2,
                 emb_scale=10,
                 emb_smooth=True,
                 # Convolutions
                 kernel_size=8,
                 strides=[4, 4, 2, 2],
                 context=1,
                 context_enc=0,
                 freq_ends=4,
                 enc_freq_attn=4,
                 # Normalization
                 norm_starts=2,
                 norm_groups=4,
                 # DConv residual branch
                 dconv_mode=1,
                 dconv_depth=2,
                 dconv_comp=4,
                 dconv_time_attn=2,
                 dconv_lstm=2,
                 dconv_init=1e-3,
                 # Weight init
                 rescale=0.1,
                 # Metadata
                 lr_sr=4000,
                 hr_sr=16000,
                 spec_upsample=True,
                 act_func='snake',
                 debug=False):
        """
        Args:
            sources (list[str]): list of source names.
            audio_channels (int): input/output audio channels.
            channels (int): initial number of hidden channels.
            growth: increase the number of hidden channels by this factor at each layer.
            nfft: number of fft bins. Note that changing this require careful computation of
                various shape parameters and will not work out of the box for hybrid models.
            end_iters: same but at train time. For a hybrid model, must be equal to `wiener_iters`.
            cac: uses complex as channels, i.e. complex numbers are 2 channels each
                in input and output. no further processing is done before ISTFT.
            depth (int): number of layers in the encoder and in the decoder.
            rewrite (bool): add 1x1 convolution to each layer.
            hybrid (bool): make a hybrid time/frequency domain, otherwise frequency only.
            hybrid_old: some models trained for MDX had a padding bug. This replicates
                this bug to avoid retraining them.
            freq_emb: add frequency embedding after the first frequency layer if > 0,
                the actual value controls the weight of the embedding.
            emb_scale: equivalent to scaling the embedding learning rate
            emb_smooth: initialize the embedding with a smooth one (with respect to frequencies).
            kernel_size: kernel_size for encoder and decoder layers.
            stride: stride for encoder and decoder layers.
            context: context for 1x1 conv in the decoder.
            context_enc: context for 1x1 conv in the encoder.
            norm_starts: layer at which group norm starts being used.
                decoder layers are numbered in reverse order.
            norm_groups: number of groups for group norm.
            dconv_mode: if 1: dconv in encoder only, 2: decoder only, 3: both.
            dconv_depth: depth of residual DConv branch.
            dconv_comp: compression of DConv branch.
            enc_freq_attn: adds freq attention layers in DConv branch starting at this layer.
            dconv_time_attn: adds time attention layers in DConv branch starting at this layer.
            dconv_lstm: adds a LSTM layer in DConv branch starting at this layer.
            dconv_init: initial scale for the DConv branch LayerScale.
            rescale: weight recaling trick
            lr_sr: source low-resolution sample-rate
            hr_sr: target high-resolution sample-rate
            spec_upsample: if true, upsamples in the spectral domain, otherwise performs sinc-interpolation beforehand
            act_func: 'snake'/'relu'
            debug: if true, prints out input dimensions throughout model layers.
        """
        super().__init__()
        self.cac = cac
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.audio_channels = audio_channels
        self.kernel_size = kernel_size
        self.context = context
        self.strides = strides
        self.depth = len(strides)
        self.channels = channels
        self.lr_sr = lr_sr
        self.hr_sr = hr_sr
        self.spec_upsample = spec_upsample

        self.scale = hr_sr / lr_sr if self.spec_upsample else 1

        self.nfft = nfft
        self.hop_length = int(hop_length // self.scale)  # this is for the input signal
        self.win_length = int(self.nfft // self.scale)  # this is for the input signal
        self.end_iters = end_iters
        self.freq_emb = None
        self.hybrid = hybrid
        self.hybrid_old = hybrid_old
        self.debug = debug

        self.encoder = nn.ModuleList()
        self.decoder = nn.ModuleList()

        chin_z = self.in_channels
        if self.cac:
            chin_z *= 2
        chout_z = channels
        freqs = nfft // 2

        for index in range(self.depth):
            freq_attn = index >= enc_freq_attn
            lstm = index >= dconv_lstm
            time_attn = index >= dconv_time_attn
            norm = index >= norm_starts
            freq = index <= freq_ends
            stri = strides[index]
            ker = kernel_size

            pad = True
            if freq and freqs < kernel_size:
                ker = freqs

            kw = {
                'kernel_size': ker,
                'stride': stri,
                'freq': freq,
                'pad': pad,
                'norm': norm,
                'rewrite': rewrite,
                'norm_groups': norm_groups,
                'dconv_kw': {
                    'lstm': lstm,
                    'time_attn': time_attn,
                    'depth': dconv_depth,
                    'compress': dconv_comp,
                    'init': dconv_init,
                    'act_func': act_func,
                    'reshape': True,
                    'freq_dim': freqs // strides[index] if freq else freqs
                }
            }

            kw_dec = dict(kw)

            enc = HEncLayer(chin_z, chout_z,
                            dconv=dconv_mode & 1, context=context_enc,
                            is_first=index == 0, freq_attn=freq_attn, freq_dim=freqs,
                            **kw)

            self.encoder.append(enc)
            if index == 0:
                chin = self.out_channels
                chin_z = chin
                if self.cac:
                    chin_z *= 2
            dec = HDecLayer(2 * chout_z, chin_z, dconv=dconv_mode & 2,
                            last=index == 0, context=context, **kw_dec)

            self.decoder.insert(0, dec)

            chin_z = chout_z
            chout_z = int(growth * chout_z)

            if freq:
                freqs //= strides[index]

            if index == 0 and freq_emb:
                self.freq_emb = ScaledEmbedding(
                    freqs, chin_z, smooth=emb_smooth, scale=emb_scale)
                self.freq_emb_scale = freq_emb

        if rescale:
            rescale_module(self, reference=rescale)

    def _spec(self, x, scale=False):
        if np.mod(x.shape[-1], self.hop_length):
            x = F.pad(x, (0, self.hop_length - np.mod(x.shape[-1], self.hop_length)))
        hl = self.hop_length
        nfft = self.nfft
        win_length = self.win_length

        if scale:
            hl = int(hl * self.scale)
            win_length = int(win_length * self.scale)

        z = spectro(x, nfft, hl, win_length=win_length)[..., :-1, :]
        return z

    def _ispec(self, z):
        hl = int(self.hop_length * self.scale)
        win_length = int(self.win_length * self.scale)
        z = F.pad(z, (0, 0, 0, 1))
        x = ispectro(z, hl, win_length=win_length)
        return x

    def _move_complex_to_channels_dim(self, z):
        B, C, Fr, T = z.shape
        m = torch.view_as_real(z).permute(0, 1, 4, 2, 3)
        m = m.reshape(B, C * 2, Fr, T)
        return m

    def _convert_to_complex(self, x):
        """

        :param x: signal of shape [Batch, Channels, 2, Freq, TimeFrames]
        :return: complex signal of shape [Batch, Channels, Freq, TimeFrames]
        """
        out = x.permute(0, 1, 3, 4, 2)
        out = torch.view_as_complex(out.contiguous())
        return out

    def forward(self, mix, return_spec=False, return_lr_spec=False):
        x = mix
        length = x.shape[-1]

        if self.debug:
            logger.info(f'hdemucs in shape: {x.shape}')

        z = self._spec(x)
        x = self._move_complex_to_channels_dim(z)

        if self.debug:
            logger.info(f'x spec shape: {x.shape}')

        B, C, Fq, T = x.shape

        # unlike previous Demucs, we always normalize because it is easier.
        mean = x.mean(dim=(1, 2, 3), keepdim=True)
        std = x.std(dim=(1, 2, 3), keepdim=True)
        x = (x - mean) / (1e-5 + std)

        # okay, this is a giant mess I know...
        saved = []  # skip connections, freq.
        lengths = []  # saved lengths to properly remove padding, freq branch.
        for idx, encode in enumerate(self.encoder):
            lengths.append(x.shape[-1])
            inject = None
            x = encode(x, inject)
            if self.debug:
                logger.info(f'encoder {idx} out shape: {x.shape}')
            if idx == 0 and self.freq_emb is not None:
                # add frequency embedding to allow for non equivariant convolutions
                # over the frequency axis.
                frs = torch.arange(x.shape[-2], device=x.device)
                emb = self.freq_emb(frs).t()[None, :, :, None].expand_as(x)
                x = x + self.freq_emb_scale * emb

            saved.append(x)

        x = torch.zeros_like(x)
        # initialize everything to zero (signal will go through u-net skips).

        for idx, decode in enumerate(self.decoder):
            skip = saved.pop(-1)
            x = decode(x, skip, lengths.pop(-1))

            if self.debug:
                logger.info(f'decoder {idx} out shape: {x.shape}')

        # Let's make sure we used all stored skip connections.
        assert len(saved) == 0

        x = x.view(B, self.out_channels, -1, Fq, T)
        x = x * std[:, None] + mean[:, None]

        if self.debug:
            logger.info(f'post view shape: {x.shape}')

        x_spec_complex = self._convert_to_complex(x)

        if self.debug:
            logger.info(f'x_spec_complex shape: {x_spec_complex.shape}')

        x = self._ispec(x_spec_complex)

        if self.debug:
            logger.info(f'hdemucs out shape: {x.shape}')

        x = x[..., :int(length * self.scale)]

        if self.debug:
            logger.info(f'hdemucs out - trimmed shape: {x.shape}')

        if return_spec:
            if return_lr_spec:
                return x, x_spec_complex, z
            else:
                return x, x_spec_complex

        return x