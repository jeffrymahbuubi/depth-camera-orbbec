import torch
import math
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple
import torchvision.transforms.functional as tvf
import random

class RandomShortSideScale(nn.Module):
    """
    Randomly pick a short‐side target in [min_size, max_size] and scale accordingly.
    Input: (T, C, H, W)
    """
    def __init__(self, min_size: int, max_size: int, interpolation: str = "bilinear"):
        super().__init__()
        self.min_size     = min_size
        self.max_size     = max_size
        self.interpolation = interpolation

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, C, H, W)
        t, c, h, w = x.shape
        size = random.randint(self.min_size, self.max_size)
        if w < h:
            new_h = int(math.floor((h / w) * size))
            new_w = size
        else:
            new_h = size
            new_w = int(math.floor((w / h) * size))
        # F.interpolate treats first dim (T) as batch:
        return F.interpolate(x, size=(new_h, new_w), 
                             mode=self.interpolation, align_corners=False)

class Normalize(nn.Module):
    """
    Normalize a (T, C, H, W) tensor by per-channel mean/std,
    treating T as the batch dimension.
    """
    def __init__(self,
                 mean: Tuple[float, float, float],
                 std:  Tuple[float, float, float],
                 inplace: bool = False):
        super().__init__()
        self.mean    = mean
        self.std     = std
        self.inplace = inplace

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x is (T, C, H, W); tvf.normalize expects (N, C, H, W)
        return tvf.normalize(x, self.mean, self.std, self.inplace)

class UniformTemporalSubsample(nn.Module):
    """
    Uniformly subsample `num_samples` frames from a (T, C, H, W) tensor.
    """
    def __init__(self, num_samples: int):
        super().__init__()
        if num_samples <= 0:
            raise ValueError(f"num_samples must be > 0, got {num_samples}")
        self.num_samples = num_samples

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (T, C, H, W)
        if x.ndim != 4:
            return x
        t = x.shape[0]
        if t == 0:
            return x
        idx = torch.linspace(0, t - 1, self.num_samples, 
                             device=x.device).round().long()
        return x.index_select(dim=0, index=idx)
