import functools
import logging
import time
import numpy as np
import cv2
import math
import torch

from torch.nn import functional as F

from contextlib import contextmanager


def get_network_description(network):
    '''Get the string and total parameters of the network'''
    if isinstance(network, torch.nn.DataParallel):
        network = network.module
    s = str(network)
    n = sum(map(lambda x: x.numel(), network.parameters()))
    return s, n

def print_network(network_name, network, logger):
    s, n = get_network_description(network)
    if isinstance(network, torch.nn.DataParallel):
        net_struc_str = '{} - {}'.format(network.__class__.__name__,
                                         network.module.__class__.__name__)
    else:
        net_struc_str = '{}'.format(network.__class__.__name__)

    logger.info(
        '{} structure: {}, with parameters: {:,d}'.format(network_name, net_struc_str, n))
    logger.info(s)


def capture_init(init):
    """capture_init.

    Decorate `__init__` with this, and you can then
    recover the *args and **kwargs passed to it in `self._init_args_kwargs`
    """

    @functools.wraps(init)
    def __init__(self, *args, **kwargs):
        self._init_args_kwargs = (args, kwargs)
        init(self, *args, **kwargs)

    return __init__


def unfold(a, kernel_size, stride):
    """Given input of size [*OT, T], output Tensor of size [*OT, F, K]
    with K the kernel size, by extracting frames with the given stride.
    This will pad the input so that `F = ceil(T / K)`.
    see https://github.com/pytorch/pytorch/issues/60466
    """
    *shape, length = a.shape
    n_frames = math.ceil(length / stride)
    tgt_length = (n_frames - 1) * stride + kernel_size
    a = F.pad(a, (0, tgt_length - length))
    strides = list(a.stride())
    assert strides[-1] == 1, 'data should be contiguous'
    strides = strides[:-1] + [stride, 1]
    return a.as_strided([*shape, n_frames, kernel_size], strides)


class LogProgress:
    """
    Sort of like tqdm but using log lines and not as real time.
    Args:
        - logger: logger obtained from `logging.getLogger`,
        - iterable: iterable object to wrap
        - updates (int): number of lines that will be printed, e.g.
            if `updates=5`, log every 1/5th of the total length.
        - total (int): length of the iterable, in case it does not support
            `len`.
        - name (str): prefix to use in the log.
        - level: logging level (like `logging.INFO`).
    """

    def __init__(self,
                 logger,
                 iterable,
                 updates=5,
                 total=None,
                 name="LogProgress",
                 level=logging.INFO):
        self.iterable = iterable
        self.total = total or len(iterable)
        self.updates = updates
        self.name = name
        self.logger = logger
        self.level = level

    def update(self, **infos):
        self._infos = infos

    def __iter__(self):
        self._iterator = iter(self.iterable)
        self._index = -1
        self._infos = {}
        self._begin = time.time()
        return self

    def __next__(self):
        self._index += 1
        try:
            value = next(self._iterator)
        except StopIteration:
            raise
        else:
            return value
        finally:
            log_every = max(1, self.total // self.updates)
            # logging is delayed by 1 it, in order to have the metrics from update
            if self._index >= 1 and self._index % log_every == 0:
                self._log()

    def _log(self):
        self._speed = (1 + self._index) / (time.time() - self._begin)
        infos = " | ".join(f"{k.capitalize()} {v}" for k, v in self._infos.items())
        if self._speed < 1e-4:
            speed = "oo sec/it"
        elif self._speed < 0.1:
            speed = f"{1 / self._speed:.1f} sec/it"
        else:
            speed = f"{self._speed:.1f} it/sec"
        out = f"{self.name} | {self._index}/{self.total} | {speed}"
        if infos:
            out += " | " + infos
        self.logger.log(self.level, out)


def scale_minmax(X, min=0.0, max=1.0):
    isnan = np.isnan(X).any()
    isinf = np.isinf(X).any()
    if isinf:
        X[X == np.inf] = 1e9
        X[X == -np.inf] = 1e-9
    if isnan:
        X[X == np.nan] = 1e-9
    # logger.info(f'isnan: {isnan}, isinf: {isinf}, max: {X.max()}, min: {X.min()}')

    X_std = (X - X.min()) / (X.max() - X.min())
    X_scaled = X_std * (max - min) + min
    return X_scaled


def convert_spectrogram_to_heatmap(spectrogram):
    spectrogram += 1e-9
    spectrogram = scale_minmax(spectrogram, 0, 255).astype(np.uint8).squeeze()
    spectrogram = np.flip(spectrogram, axis=0)
    spectrogram = 255 - spectrogram
    # spectrogram = (255 * (spectrogram - np.min(spectrogram)) / np.ptp(spectrogram)).astype(np.uint8).squeeze()[::-1,:]
    heatmap = cv2.applyColorMap(spectrogram, cv2.COLORMAP_INFERNO)
    heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
    return heatmap


def colorize(text, color):
    """
    Display text with some ANSI color in the terminal.
    """
    code = f"\033[{color}m"
    restore = "\033[0m"
    return "".join([code, text, restore])


def bold(text):
    """
    Display text in bold in the terminal.
    """
    return colorize(text, "1")


def copy_state(state):
    return {k: v.cpu().clone() for k, v in state.items()}


def serialize_model(model):
    args, kwargs = model._init_args_kwargs
    state = copy_state(model.state_dict())
    return {"class": model.__class__, "args": args, "kwargs": kwargs, "state": state}


@contextmanager
def swap_state(model, state):
    """
    Context manager that swaps the state of a model, e.g:

        # model is in old state
        with swap_state(model, new_state):
            # model in new state
        # model back to old state
    """
    old_state = copy_state(model.state_dict())
    model.load_state_dict(state)
    try:
        yield
    finally:
        model.load_state_dict(old_state)


def pull_metric(history, name):
    out = []
    for metrics in history:
        if name in metrics:
            out.append(metrics[name])
    return out


def match_signal(signal, ref_len):
    sig_len = signal.shape[-1]
    if sig_len < ref_len:
        signal = F.pad(signal, (0, ref_len - sig_len))
    elif sig_len > ref_len:
        signal = signal[..., :ref_len]
    return signal