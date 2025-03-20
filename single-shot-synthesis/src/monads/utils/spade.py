import typing

import numpy as np
import torch
import src.components.ganimator as ganimator

__all__ = ["Spade"]


class Spade(torch.nn.Module):
    def __init__(
        self,
        parents: typing.Sequence[int],
        contacts: typing.Sequence[int],
        kernel_size: int,
        padding_mode="reflect",
        bias=True,
    ) -> None:
        super().__init__()
        padding = (kernel_size - 1) // 2
        neighbors = ganimator.get_neighbor(parents, contacts, threshold=2, enforce_contact=True)
        num_features = (len(parents) + len(contacts) + 1) * 6
        channels = ganimator.get_channels_list(num_features)
        
        self.gamma = ganimator.SkeletonBlock(
            neighbors,
            channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            padding=padding,
            bias=bias,
            activation_type="lrelu",
        )

        self.beta = ganimator.SkeletonBlock(
            neighbors,
            channels,
            kernel_size=kernel_size,
            padding_mode=padding_mode,
            padding=padding,
            bias=bias,
            activation_type="lrelu",
        )

    def forward(
        self,
        x: torch.Tensor,
        skeleton_id_map: torch.Tensor,
    ) -> torch.Tensor:
        gamma = self.gamma(skeleton_id_map)
        beta = self.beta(skeleton_id_map)
        return x * (1 + gamma) + beta
            
        

        
        

