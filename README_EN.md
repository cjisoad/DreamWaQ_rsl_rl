# DreamWaQ-RSL-RL

[中文说明](./README.md)

This repository is a lightweight reinforcement learning library for robotics, developed on top of the original `rsl_rl`, with added support for the **DreamWaQ** algorithm while preserving the native calling style.

The DreamWaQ integration is designed to stay as close as possible to native `rsl_rl`, so that Isaac Lab can be adapted mainly through configuration changes.

Validated with:

- `Isaac Sim 5.1.0`
- `Isaac Lab 2.2.1`

## Repository Contents

The current algorithm library includes:

- `PPO`
- `Student-Teacher Distillation`
- `DWAQPPO`
- `Random Network Distillation (RND)`
- `Symmetry-based Augmentation`

DreamWaQ-related entry points:

- [`rsl_rl/algorithms/dwaq_ppo.py`](./rsl_rl/algorithms/dwaq_ppo.py)
- [`rsl_rl/modules/actor_critic_DWAQ.py`](./rsl_rl/modules/actor_critic_DWAQ.py)
- [`rsl_rl/runners/on_policy_runner.py`](./rsl_rl/runners/on_policy_runner.py)

## Installation

Recommended installation:

```bash
git clone <your-repo-url>
cd rsl_rl
pip install -e .
```

## Calling Flow

The standard `rsl_rl` flow is:

1. the environment returns a `TensorDict`
2. `OnPolicyRunner` resolves `obs_groups`
3. the policy extracts actor / critic inputs from the `TensorDict`
4. the algorithm runs `act / process_env_step / compute_returns / update`

This is also how DreamWaQ works in this repository. In other words, DreamWaQ no longer needs a dedicated training branch.

## DreamWaQ Integration Guide

### 1. Observation protocol

To use DreamWaQ, the `TensorDict` returned by the environment should contain at least:

- `policy`
- `obs_history` or `obs_hist`
- one or more critic-related groups such as `privileged`

Typical structure:

```python
from tensordict import TensorDict

obs = TensorDict(
    {
        "policy": actor_obs,
        "privileged": privileged_obs,
        "obs_history": obs_history,
    },
    batch_size=[num_envs],
)
```

Where:

- `policy` is the current actor observation
- `obs_history` is the flattened observation history
- critic inputs are defined by `obs_groups["critic"]`

### 2. `obs_groups`

DreamWaQ still uses the native `obs_groups` mechanism:

```yaml
obs_groups: {"policy": ["policy"], "critic": ["policy", "privileged"]}
```

This means:

- the actor consumes `policy`
- the critic consumes `policy + privileged`

### 3. Important Constraint

The current DreamWaQ velocity supervision still assumes:

```python
vel_target = critic_obs[:, obs_dim : obs_dim + 3]
```

So the critic input layout must satisfy:

1. the first `obs_dim` dimensions match the actor observation
2. the next `3` dimensions correspond to target velocity

If your critic observation layout differs, you need to adjust the DreamWaQ loss accordingly.

## DreamWaQ Configuration Guide

Below is a minimal configuration snippet. The caller-side style remains very close to native PPO.

```yaml
runner:
  class_name: OnPolicyRunner
  num_steps_per_env: 24
  max_iterations: 1500
  save_interval: 50
  logger: tensorboard
  seed: 1
  obs_groups: {"policy": ["policy"], "critic": ["policy", "privileged"]}

  policy:
    class_name: ActorCritic_DWAQ
    activation: elu
    actor_hidden_dims: [512, 256, 128]
    critic_hidden_dims: [512, 256, 128]
    init_noise_std: 1.0
    cenet_out_dim: 19
    actor_obs_normalization: false
    critic_obs_normalization: false

  algorithm:
    class_name: DWAQPPO
    learning_rate: 0.001
    num_learning_epochs: 5
    num_mini_batches: 4
    clip_param: 0.2
    gamma: 0.99
    lam: 0.95
    value_loss_coef: 1.0
    entropy_coef: 0.01
    max_grad_norm: 1.0
    use_clipped_value_loss: true
    desired_kl: 0.01
    schedule: adaptive
    beta: 1.0
```

The project still keeps the `DWAQOnPolicyRunner` class name for compatibility. It can still be used, but `OnPolicyRunner` is the recommended entry point.

## Minimal DreamWaQ Environment Example

```python
import torch
from tensordict import TensorDict


class MyEnv:
    def __init__(self, num_envs=4096, device="cuda:0"):
        self.num_envs = num_envs
        self.num_actions = 12
        self.device = device
        self.max_episode_length = 1000
        self.episode_length_buf = torch.zeros(num_envs, device=device, dtype=torch.long)

    def get_observations(self):
        actor_obs = torch.zeros(self.num_envs, 45, device=self.device)
        privileged_obs = torch.zeros(self.num_envs, 48, device=self.device)
        obs_history = torch.zeros(self.num_envs, 45 * 5, device=self.device)
        return TensorDict(
            {
                "policy": actor_obs,
                "privileged": privileged_obs,
                "obs_history": obs_history,
            },
            batch_size=[self.num_envs],
        )

    def step(self, actions):
        obs = self.get_observations()
        rewards = torch.zeros(self.num_envs, device=self.device)
        dones = torch.zeros(self.num_envs, device=self.device, dtype=torch.bool)
        extras = {"time_outs": torch.zeros(self.num_envs, device=self.device)}
        return obs, rewards, dones, extras
```

## Minimal Training Example

```python
from rsl_rl.runners import OnPolicyRunner

env = MyEnv()
runner_cfg = train_cfg["runner"]
runner = OnPolicyRunner(env, runner_cfg, log_dir="logs/demo", device="cuda:0")
runner.learn(num_learning_iterations=runner_cfg["max_iterations"])
```

## Credits

The original `rsl_rl` is maintained by ETH Zurich Robotic Systems Lab and NVIDIA. See:

- [CONTRIBUTORS.md](./CONTRIBUTORS.md)
- [CITATION.cff](./CITATION.cff)
