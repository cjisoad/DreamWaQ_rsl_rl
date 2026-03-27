# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from torch.distributions import Normal


class ActorCritic_DWAQ(nn.Module):
    """Actor-critic with a beta-VAE context encoder used by DWAQ."""

    is_recurrent: bool = False

    def __init__(
        self,
        num_actor_obs: int,
        num_critic_obs: int,
        num_actions: int,
        cenet_in_dim: int,
        cenet_out_dim: int,
        obs_dim: int,
        activation: str = "elu",
        init_noise_std: float = 1.0,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_dim = obs_dim
        make_activation = _get_activation_factory(activation)

        self.actor = nn.Sequential(
            nn.Linear(num_actor_obs, 512),
            make_activation(),
            nn.Linear(512, 256),
            make_activation(),
            nn.Linear(256, 128),
            make_activation(),
            nn.Linear(128, num_actions),
        )
        print(f"DWAQ Actor MLP: {self.actor}")

        self.critic = nn.Sequential(
            nn.Linear(num_critic_obs, 512),
            make_activation(),
            nn.Linear(512, 256),
            make_activation(),
            nn.Linear(256, 128),
            make_activation(),
            nn.Linear(128, 1),
        )
        print(f"DWAQ Critic MLP: {self.critic}")

        self.encoder = nn.Sequential(
            nn.Linear(cenet_in_dim, 128),
            make_activation(),
            nn.Linear(128, 64),
            make_activation(),
        )
        self.encode_mean_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(64, cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(64, 3)
        self.encode_logvar_vel = nn.Linear(64, 3)

        self.decoder = nn.Sequential(
            nn.Linear(cenet_out_dim, 64),
            make_activation(),
            nn.Linear(64, 128),
            make_activation(),
            nn.Linear(128, self.obs_dim),
        )

        self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    def reset(self, dones: torch.Tensor | None = None) -> None:
        pass

    def forward(self) -> NoReturn:
        raise NotImplementedError

    @property
    def action_mean(self) -> torch.Tensor:
        return self.distribution.mean

    @property
    def action_std(self) -> torch.Tensor:
        return self.distribution.stddev

    @property
    def entropy(self) -> torch.Tensor:
        return self.distribution.entropy().sum(dim=-1)

    def reparameterise(self, mean: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        logvar = torch.clamp(logvar, min=-10.0, max=10.0)
        std = torch.exp(0.5 * logvar)
        return mean + std * torch.randn_like(std)

    def cenet_forward(
        self, obs_history: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        encoded = self.encoder(obs_history)
        mean_latent = self.encode_mean_latent(encoded)
        logvar_latent = self.encode_logvar_latent(encoded)
        mean_vel = self.encode_mean_vel(encoded)
        logvar_vel = self.encode_logvar_vel(encoded)
        code_latent = self.reparameterise(mean_latent, logvar_latent)
        code_vel = self.reparameterise(mean_vel, logvar_vel)
        code = torch.cat((code_vel, code_latent), dim=-1)
        decode = self.decoder(code)
        return code, code_vel, decode, mean_vel, logvar_vel, mean_latent, logvar_latent

    def _update_distribution(self, observations: torch.Tensor) -> None:
        mean = torch.nan_to_num(self.actor(observations), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
        std = torch.clamp(self.std, min=1.0e-6, max=1.0e3).unsqueeze(0).expand_as(mean)
        if not torch.isfinite(mean).all() or not torch.isfinite(std).all():
            mean = torch.where(torch.isfinite(mean), mean, torch.zeros_like(mean))
            std = torch.where(torch.isfinite(std), std, torch.full_like(std, 1.0e-3))
        self.distribution = Normal(mean, std)

    def act(self, observations: torch.Tensor, obs_history: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        self._update_distribution(torch.cat((code, observations), dim=-1))
        return self.distribution.sample()

    def act_inference(self, observations: torch.Tensor, obs_history: torch.Tensor) -> torch.Tensor:
        code, _, _, _, _, _, _ = self.cenet_forward(obs_history)
        return self.actor(torch.cat((code, observations), dim=-1))

    def evaluate(self, critic_observations: torch.Tensor, **kwargs: dict[str, Any]) -> torch.Tensor:
        return self.critic(critic_observations)

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True


def _get_activation_factory(act_name: str) -> type[nn.Module]:
    activations = {
        "elu": nn.ELU,
        "selu": nn.SELU,
        "relu": nn.ReLU,
        "lrelu": nn.LeakyReLU,
        "tanh": nn.Tanh,
        "sigmoid": nn.Sigmoid,
    }
    try:
        return activations[act_name]
    except KeyError as exc:
        raise ValueError(f"Invalid activation function: {act_name}") from exc
