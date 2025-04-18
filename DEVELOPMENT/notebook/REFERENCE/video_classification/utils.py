import math
import random
from typing import Any, Dict, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms.functional as tvf


class ApplyTransformToKey(nn.Module):
    """
    Wrap a Tensor→Tensor transform so it only applies to sample[key] in a dict.
    """
    def __init__(self, key: str, transform: nn.Module):
        super().__init__()
        self.key = key
        self.transform = transform

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out = sample.copy()
        if self.key in out:
            out[self.key] = self.transform(out[self.key])
        return out


class RemoveKey(nn.Module):
    """Remove sample[key] from the dict."""
    def __init__(self, key: str):
        super().__init__()
        self.key = key

    def forward(self, sample: Dict[str, Any]) -> Dict[str, Any]:
        out = sample.copy()
        out.pop(self.key, None)
        return out


class ShortSideScale(nn.Module):
    """
    Scale the shorter spatial side of a (C, T, H, W) tensor to `size`,
    preserving aspect ratio.
    """
    def __init__(self, size: int, interpolation: str = "bilinear"):
        super().__init__()
        self.size = size
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T, H, W)
        if x.ndim != 4:
            return x
        c, t, h, w = x.shape
        if w < h:
            new_h = int(math.floor((h / w) * self.size))
            new_w = self.size
        else:
            new_h = self.size
            new_w = int(math.floor((w / h) * self.size))
        return F.interpolate(x, size=(new_h, new_w), mode=self.interpolation, align_corners=False)


class RandomShortSideScale(nn.Module):
    """
    Randomly pick a short-side target in [min_size, max_size] and scale accordingly.
    """
    def __init__(self, min_size: int, max_size: int, interpolation: str = "bilinear"):
        super().__init__()
        self.min_size = min_size
        self.max_size = max_size
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T, H, W)
        size = random.randint(self.min_size, self.max_size)
        c, t, h, w = x.shape
        if w < h:
            new_h = int(math.floor((h / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((w / h) * size))
        return F.interpolate(x, size=(new_h, new_w), mode=self.interpolation, align_corners=False)


class Normalize(nn.Module):
    """
    Normalize a (C, T, H, W) tensor by per-channel mean/std.
    """
    def __init__(self, mean: Tuple[float, float, float], std: Tuple[float, float, float], inplace: bool = False):
        super().__init__()
        self.mean = mean
        self.std = std
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # tvf.normalize expects (T, C, H, W)
        vid = x.permute(1, 0, 2, 3)
        vid = tvf.normalize(vid, self.mean, self.std, self.inplace)
        return vid.permute(1, 0, 2, 3)


class UniformTemporalSubsample(nn.Module):
    """
    Uniformly subsample `num_samples` frames from a (C, T, H, W) tensor.
    """
    def __init__(self, num_samples: int):
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        self.num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (C, T, H, W)
        if x.ndim != 4:
            return x
        t = x.shape[1]
        if t == 0:
            return x
        idx = torch.linspace(0, t - 1, self.num_samples, device=x.device).round().long()
        return x.index_select(dim=1, index=idx)
