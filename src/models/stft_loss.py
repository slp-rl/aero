"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""

"""STFT-based Loss modules."""

import torch
import torch.nn.functional as F


def stft(x, fft_size, hop_size, win_length, window, start_interval=0.0, end_interval=1.0):
    """Perform STFT and convert to magnitude spectrogram.
    Args:
        x (Tensor): Input signal tensor (B, T).
        fft_size (int): FFT size.
        hop_size (int): Hop size.
        win_length (int): Window length.
        window (str): Window function type.
        start_interval (float): where to start (between 0 hz and the nyquist frequency)
        end_interval (float): where to end (between 0 hz and the nyquist frequency)
    Returns:
        Tensor: Magnitude spectrogram (B, #frames, fft_size // 2 + 1).
    """
    x_stft = torch.view_as_real(torch.stft(x, fft_size, hop_size, win_length, window, return_complex=True))
    real = x_stft[..., 0]
    imag = x_stft[..., 1]

    if start_interval > 0:
        real[:,0:int(start_interval*real.shape[1])-1,:] = 0
        imag[:,0:int(start_interval*real.shape[1])-1,:] = 0
    
    if end_interval < 1:
        real[:,int(end_interval*real.shape[1])-1:,:] = 0
        imag[:,int(end_interval*real.shape[1])-1:,:] = 0

    # NOTE(kan-bayashi): clamp is needed to avoid nan or inf
    return torch.sqrt(torch.clamp(real ** 2 + imag ** 2, min=1e-7)).transpose(2, 1), torch.atan2(imag, real).transpose(2, 1)


class SpectralConvergengeLoss(torch.nn.Module):
    """Spectral convergence loss module."""

    def __init__(self, magnitude_weight_shift=0.0):
        """Initilize spectral convergence loss module."""
        super(SpectralConvergengeLoss, self).__init__()
        self.freq_weights = None
        self.magnitude_weight_shift = magnitude_weight_shift

    def forward(self, x_mag, y_mag, x_phase, y_phase):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #freq_bins, #frames).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #freq_bins, #frames).
            x_phase (Tensor): Phase spectrogram of predicted signal (B, #freq_bins, #frames).
            y_phase (Tensor): Phase spectrogram of groundtruth signal (B, #freq_bins, #frames).
        Returns:
            Tensor: Spectral convergence loss value.
        """
        if self.magnitude_weight_shift != 0.0:
            mag_diff = torch.where(y_mag > x_mag,
                                   (1+self.magnitude_weight_shift)*(y_mag - x_mag), 
                                   (1-self.magnitude_weight_shift)*(y_mag - x_mag)
                                   )
        else:
            mag_diff = y_mag - x_mag
        mag_conv_loss = torch.norm(mag_diff, p="fro") / torch.norm(y_mag, p="fro")
        phase_conv_loss = torch.mean(2* torch.asin(torch.sqrt(torch.sin((x_phase - y_phase)/2)**2)))
        return mag_conv_loss + (phase_conv_loss.item()/5)


class LogSTFTMagnitudeLoss(torch.nn.Module):
    """Log STFT magnitude loss module."""

    def __init__(self, magnitude_weight_shift=0.0):
        """Initilize los STFT magnitude loss module."""
        super(LogSTFTMagnitudeLoss, self).__init__()
        self.magnitude_weight_shift = magnitude_weight_shift

    def forward(self, x_mag, y_mag):
        """Calculate forward propagation.
        Args:
            x_mag (Tensor): Magnitude spectrogram of predicted signal (B, #frames, #freq_bins).
            y_mag (Tensor): Magnitude spectrogram of groundtruth signal (B, #frames, #freq_bins).
        Returns:
            Tensor: Log STFT magnitude loss value.
        """
        if self.magnitude_weight_shift != 0.0:
            log_x_mag = torch.log(x_mag)
            log_y_mag = torch.log(y_mag)
            log_mag_diff = torch.abs(torch.where(log_y_mag > log_x_mag,
                                   (1+self.magnitude_weight_shift)*(log_y_mag - log_x_mag), 
                                   (1-self.magnitude_weight_shift)*(log_y_mag - log_x_mag)
                                   ))
            return torch.mean(log_mag_diff)
        else:
            return F.l1_loss(torch.log(y_mag), torch.log(x_mag))


class STFTLoss(torch.nn.Module):
    """STFT loss module."""

    def __init__(self, fft_size=1024, shift_size=120, win_length=600, window="hann_window", start_interval=0.0, end_interval=1.0, magnitude_weight_shift=0.0):
        """Initialize STFT loss module."""
        super(STFTLoss, self).__init__()
        self.fft_size = fft_size
        self.shift_size = shift_size
        self.win_length = win_length
        self.register_buffer("window", getattr(torch, window)(win_length))
        self.magnitude_weight_shift = magnitude_weight_shift
        self.spectral_convergenge_loss = SpectralConvergengeLoss(magnitude_weight_shift)
        self.log_stft_magnitude_loss = LogSTFTMagnitudeLoss(magnitude_weight_shift)
        self.start_interval = start_interval
        self.end_interval = end_interval

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Spectral convergence loss value.
            Tensor: Log STFT magnitude loss value.
        """
        x_mag, x_phase = stft(x, self.fft_size, self.shift_size, self.win_length, self.window, self.start_interval, self.end_interval)
        y_mag, y_phase = stft(y, self.fft_size, self.shift_size, self.win_length, self.window, self.start_interval, self.end_interval)
        sc_loss = self.spectral_convergenge_loss(x_mag, y_mag, x_phase, y_phase)
        mag_loss = self.log_stft_magnitude_loss(x_mag, y_mag)

        return sc_loss, mag_loss


class MultiResolutionSTFTLoss(torch.nn.Module):
    """Multi resolution STFT loss module."""

    def __init__(self,
                 fft_sizes=[1531, 2557, 4099, 4999, 3001, 6037],
                 hop_sizes=[127, 307, 241, 401, 547, 577],
                 win_lengths=[601, 1399, 1249, 4999, 1601, 5167],
                 window="hann_window", factor_sc=0.1, factor_mag=0.1,
                 start_interval=0.0, end_interval=1.0, magnitude_weight_shift=0.0):
        """Initialize Multi resolution STFT loss module.
        Args:
            fft_sizes (list): List of FFT sizes.
            hop_sizes (list): List of hop sizes.
            win_lengths (list): List of window lengths.
            window (str): Window function type.
            factor (float): a balancing factor across different losses.
            start_interval (float): where to start (between 0 hz and the nyquist frequency)
            end_interval (float): where to end (between 0 hz and the nyquist frequency)
            magnitude_weight_shift (float): if non-zero, increases loss for predicted magnititudes that are too low (positive) or high (negative)
        """
        super(MultiResolutionSTFTLoss, self).__init__()
        assert len(fft_sizes) == len(hop_sizes) == len(win_lengths)
        assert magnitude_weight_shift > -1.0, magnitude_weight_shift < 1.0
        self.stft_losses = torch.nn.ModuleList()
        for fs, ss, wl in zip(fft_sizes, hop_sizes, win_lengths):
            self.stft_losses += [STFTLoss(fs, ss, wl, window, start_interval, end_interval, magnitude_weight_shift)]
        self.factor_sc = factor_sc
        self.factor_mag = factor_mag

    def forward(self, x, y):
        """Calculate forward propagation.
        Args:
            x (Tensor): Predicted signal (B, T).
            y (Tensor): Groundtruth signal (B, T).
        Returns:
            Tensor: Multi resolution spectral convergence loss value.
            Tensor: Multi resolution log STFT magnitude loss value.
        """
        sc_loss = 0.0
        mag_loss = 0.0
        for f in self.stft_losses:
            sc_l, mag_l = f(x, y)
            sc_loss += sc_l
            mag_loss += mag_l
        sc_loss /= len(self.stft_losses)
        mag_loss /= len(self.stft_losses)

        return self.factor_sc*sc_loss, self.factor_mag*mag_loss
