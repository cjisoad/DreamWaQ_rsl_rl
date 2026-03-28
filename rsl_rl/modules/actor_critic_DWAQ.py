# Copyright (c) 2021-2025, ETH Zurich and NVIDIA CORPORATION
# All rights reserved.
#
# SPDX-License-Identifier: BSD-3-Clause

from __future__ import annotations

from typing import Any, NoReturn

import torch
import torch.nn as nn
from tensordict import TensorDict
from torch.distributions import Normal

from rsl_rl.networks import EmpiricalNormalization, MLP


class ActorCritic_DWAQ(nn.Module):
    """Actor-critic with a beta-VAE context encoder used by DWAQ."""

    is_recurrent: bool = False

    def __init__(
        self,
        obs: TensorDict,
        obs_groups: dict[str, list[str]],
        num_actions: int,
        actor_obs_normalization: bool = False,
        critic_obs_normalization: bool = False,
        actor_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        critic_hidden_dims: tuple[int] | list[int] = [512, 256, 128],
        activation: str = "elu",
        init_noise_std: float = 1.0,
        noise_std_type: str = "scalar",
        state_dependent_std: bool = False,
        cenet_hidden_dims: tuple[int] | list[int] = [128, 64],
        cenet_out_dim: int = 19,
        obs_history_key: str = "obs_history",
        obs_dim: int | None = None,
        **kwargs: dict[str, Any],
    ) -> None:
        if kwargs:
            print(
                "ActorCritic_DWAQ.__init__ got unexpected arguments, which will be ignored: "
                + str([key for key in kwargs])
            )
        super().__init__()

        self.obs_groups = obs_groups
        self.obs_history_key = self._resolve_obs_history_key(obs, obs_history_key)
        self.state_dependent_std = state_dependent_std

        num_actor_obs = 0
        for obs_group in obs_groups["policy"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic_DWAQ module only supports 1D observations."
            num_actor_obs += obs[obs_group].shape[-1]

        num_critic_obs = 0
        for obs_group in obs_groups["critic"]:
            assert len(obs[obs_group].shape) == 2, "The ActorCritic_DWAQ module only supports 1D observations."
            num_critic_obs += obs[obs_group].shape[-1]

        obs_history_dim = obs[self.obs_history_key].shape[-1]
        self.obs_dim = num_actor_obs if obs_dim is None else obs_dim

        if self.state_dependent_std:
            self.actor = MLP(num_actor_obs + cenet_out_dim, [2, num_actions], actor_hidden_dims, activation)
        else:
            self.actor = MLP(num_actor_obs + cenet_out_dim, num_actions, actor_hidden_dims, activation)
        print(f"DWAQ Actor MLP: {self.actor}")

        self.critic = MLP(num_critic_obs, 1, critic_hidden_dims, activation)
        print(f"DWAQ Critic MLP: {self.critic}")

        self.actor_obs_normalization = actor_obs_normalization
        if actor_obs_normalization:
            self.actor_obs_normalizer = EmpiricalNormalization(num_actor_obs)
        else:
            self.actor_obs_normalizer = torch.nn.Identity()

        self.critic_obs_normalization = critic_obs_normalization
        if critic_obs_normalization:
            self.critic_obs_normalizer = EmpiricalNormalization(num_critic_obs)
        else:
            self.critic_obs_normalizer = torch.nn.Identity()

        self.encoder = MLP(obs_history_dim, cenet_hidden_dims[-1], cenet_hidden_dims[:-1], activation)
        self.encode_mean_latent = nn.Linear(cenet_hidden_dims[-1], cenet_out_dim - 3)
        self.encode_logvar_latent = nn.Linear(cenet_hidden_dims[-1], cenet_out_dim - 3)
        self.encode_mean_vel = nn.Linear(cenet_hidden_dims[-1], 3)
        self.encode_logvar_vel = nn.Linear(cenet_hidden_dims[-1], 3)
        self.decoder = MLP(cenet_out_dim, self.obs_dim, [64, 128], activation)

        self.noise_std_type = noise_std_type
        if self.state_dependent_std:
            torch.nn.init.zeros_(self.actor[-2].weight[num_actions:])
            if self.noise_std_type == "scalar":
                torch.nn.init.constant_(self.actor[-2].bias[num_actions:], init_noise_std)
            elif self.noise_std_type == "log":
                torch.nn.init.constant_(
                    self.actor[-2].bias[num_actions:], torch.log(torch.tensor(init_noise_std + 1e-7))
                )
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            if self.noise_std_type == "scalar":
                self.std = nn.Parameter(init_noise_std * torch.ones(num_actions))
            elif self.noise_std_type == "log":
                self.log_std = nn.Parameter(torch.log(init_noise_std * torch.ones(num_actions)))
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")

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

    def _update_distribution(self, obs: torch.Tensor) -> None:
        if self.state_dependent_std:
            mean_and_std = self.actor(obs)
            if self.noise_std_type == "scalar":
                mean, std = torch.unbind(mean_and_std, dim=-2)
            elif self.noise_std_type == "log":
                mean, log_std = torch.unbind(mean_and_std, dim=-2)
                std = torch.exp(log_std)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        else:
            mean = torch.nan_to_num(self.actor(obs), nan=0.0, posinf=1.0e6, neginf=-1.0e6)
            if self.noise_std_type == "scalar":
                std = torch.clamp(self.std, min=1.0e-6, max=1.0e3).expand_as(mean)
            elif self.noise_std_type == "log":
                std = torch.exp(self.log_std).expand_as(mean)
            else:
                raise ValueError(f"Unknown standard deviation type: {self.noise_std_type}. Should be 'scalar' or 'log'")
        self.distribution = Normal(mean, std)

    def act(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        code, _, _, _, _, _, _ = self.cenet_forward(self.get_obs_history(obs))
        self._update_distribution(torch.cat((code, actor_obs), dim=-1))
        return self.distribution.sample()

    def act_inference(self, obs: TensorDict) -> torch.Tensor:
        actor_obs = self.actor_obs_normalizer(self.get_actor_obs(obs))
        code, _, _, _, _, _, _ = self.cenet_forward(self.get_obs_history(obs))
        policy_input = torch.cat((code, actor_obs), dim=-1)
        if self.state_dependent_std:
            return self.actor(policy_input)[..., 0, :]
        return self.actor(policy_input)

    def evaluate(self, obs: TensorDict, **kwargs: dict[str, Any]) -> torch.Tensor:
        critic_obs = self.critic_obs_normalizer(self.get_critic_obs(obs))
        return self.critic(critic_obs)

    def get_actor_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["policy"]]
        return torch.cat(obs_list, dim=-1)

    def get_critic_obs(self, obs: TensorDict) -> torch.Tensor:
        obs_list = [obs[obs_group] for obs_group in self.obs_groups["critic"]]
        return torch.cat(obs_list, dim=-1)

    def get_obs_history(self, obs: TensorDict) -> torch.Tensor:
        return obs[self.obs_history_key]

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        return self.distribution.log_prob(actions).sum(dim=-1)

    def update_normalization(self, obs: TensorDict) -> None:
        if self.actor_obs_normalization:
            self.actor_obs_normalizer.update(self.get_actor_obs(obs))
        if self.critic_obs_normalization:
            self.critic_obs_normalizer.update(self.get_critic_obs(obs))

    def load_state_dict(self, state_dict: dict, strict: bool = True) -> bool:
        super().load_state_dict(state_dict, strict=strict)
        return True

    @staticmethod
    def _resolve_obs_history_key(obs: TensorDict, obs_history_key: str) -> str:
        if obs_history_key in obs:
            return obs_history_key
        if "obs_hist" in obs:
            return "obs_hist"
        raise KeyError(
            f"Observation history key '{obs_history_key}' not found in observations. Available keys: {list(obs.keys())}"
        )
