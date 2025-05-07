import typing

import numpy as np
import torch
import src.components.ganimator as ganimator

__all__ = ["FiLM"]


# class FiLM(torch.nn.Module):
#     def __init__(
#         self,
#         parents: typing.Sequence[int],
#         contacts: typing.Sequence[int],
#         bias=True,
#     ) -> None:
#         super().__init__()
#         neighbors = ganimator.get_neighbor(parents, contacts, threshold=2, enforce_contact=True)
#         num_features = (len(parents) + len(contacts) + 1) * 6
#         channels = ganimator.get_channels_list(num_features)

#         # self.gamma = ganimator.SkeletonBlockLinear(
#         #     neighbors,
#         #     channels,
#         #     bias=bias,
#         #     activation_type="lrelu",
#         # )

#         # self.beta = ganimator.SkeletonBlockLinear(
#         #     neighbors,
#         #     channels,
#         #     bias=bias,
#         #     activation_type="lrelu",
#         # )
#         self.gamma = torch.nn.Linear(174,174)
#         self.beta = torch.nn.Linear(174,174)

#     def forward(
#         self,
#         x: torch.Tensor,
#         skeleton_id_map: torch.Tensor,
#     ) -> torch.Tensor:
        
#         mod_input = skeleton_id_map.permute(0, 2, 1).reshape(-1, 174)  # [B × T, 174]
#         gamma = self.gamma(mod_input)
#         beta = self.beta(mod_input)
#         gamma = gamma.view(skeleton_id_map.shape[0], skeleton_id_map.shape[2], 174).permute(0, 2, 1)  # [B, 174, T]
#         beta = beta.view(skeleton_id_map.shape[0], skeleton_id_map.shape[2], 174).permute(0, 2, 1)    # [B, 174, T]

#         return x * gamma + beta



class FiLM(torch.nn.Module):
    def __init__(
        self,
        parents: typing.Sequence[int],
        contacts: typing.Sequence[int],
        bias=True,
    ) -> None:
        super().__init__()
        neighbors = ganimator.get_neighbor(parents, contacts, threshold=2, enforce_contact=True)
        num_features = (len(parents) + len(contacts) + 1) * 6
        channels = ganimator.get_channels_list(num_features)

        self.gamma = ganimator.SkeletonBlockLinear(
            neighbors,
            channels,
            bias=bias,
            activation_type="lrelu",
        )

        self.beta = ganimator.SkeletonBlockLinear(
            neighbors,
            channels,
            bias=bias,
            activation_type="lrelu",
        )
        # self.gamma = torch.nn.Linear(174,174)
        # self.beta = torch.nn.Linear(174,174)

    def forward(
        self,
        x: torch.Tensor,
        skeleton_id_map: torch.Tensor,
    ) -> torch.Tensor:
        
        mod_input = skeleton_id_map.permute(0, 2, 1).reshape(-1, 174)  # [B × T, 174]
        gamma = self.gamma(mod_input)
        beta = self.beta(mod_input)
        gamma = gamma.view(skeleton_id_map.shape[0], skeleton_id_map.shape[2], 174).permute(0, 2, 1)  # [B, 174, T]
        beta = beta.view(skeleton_id_map.shape[0], skeleton_id_map.shape[2], 174).permute(0, 2, 1)    # [B, 174, T]

        return x * gamma + beta
            
        

        
        

