"""
This code is based on Facebook's HDemucs code: https://github.com/facebookresearch/demucs
"""
import math
import torchaudio
from torch.nn import functional as F


class Audioset:
    def __init__(self, files=None, length=None, stride=None,
                 pad=True, with_path=False, sample_rate=None,
                 channels=None, stride_offset_ratio = 0):
        """
        files should be a list [(file, length)]
        """
        self.files = files
        self.num_examples = []
        self.length = length
        self.stride = stride or length
        self.with_path = with_path
        self.sample_rate = sample_rate
        self.channels = channels
        self.stride_offset =  int(stride_offset_ratio * stride) if (stride != length) else 0

        widest_range = max(self.stride, self.length)

        for file, file_length in self.files:
            if length is None:
                examples = 1
            elif file_length < length:
                examples = 1 if pad else 0
            elif pad:
                examples = int(math.ceil((file_length - widest_range) / widest_range) - 1)
            else:
                examples = (file_length - widest_range) // widest_range - 1
            self.num_examples.append(examples)

    def __len__(self):
        return sum(self.num_examples)

    def __getitem__(self, index):
        for (file, _), examples in zip(self.files, self.num_examples):
            if index >= examples:
                index -= examples
                continue
            num_frames = 0
            offset = 0
            if self.length is not None:
                offset = self.stride * index + self.stride_offset
                num_frames = self.length
            out, sr = torchaudio.load(str(file),
                                        frame_offset=offset,
                                        num_frames=num_frames or -1)

            if sr != self.sample_rate:
                raise RuntimeError(f"Expected {file} to have sample rate of "
                                   f"{self.sample_rate}, but got {sr}")
            if out.shape[0] != self.channels:
                raise RuntimeError(f"Expected {file} to have shape of "
                                   f"{self.channels}, but got {out.shape[0]}")
            if num_frames:
                out = F.pad(out, (0, num_frames - out.shape[-1]))
            if self.with_path:
                return out, file
            else:
                return out
            
    def update_stride_offset(self, new_ratio):
        assert(new_ratio >= 0 and new_ratio <= 1)
        self.stride_offset = int(new_ratio * self.stride) if (self.stride != self.length) else 0