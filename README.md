# DreamWaQ-RSL-RL

[English Version](./README_EN.md)

面向机器人强化学习的轻量训练库。本仓库基于原始 `rsl_rl` 进行开发，并在保持原生调用风格的前提下加入了**DreamWaQ**算法支持。

DreamWaQ算法的调用尽量沿用原生 `rsl_rl` 的风格，确保在只改动配置文件的基础上即可适配IsaacLab：

已使用以下版本的软件验证：

- `Isaac Sim 5.1.0`
- `Isaac Lab 2.2.1`

## 仓库内容

当前算法库包含以下算法与组件：

- `PPO`
- `Student-Teacher Distillation`
- `DWAQPPO`
- `Random Network Distillation (RND)`
- `Symmetry-based Augmentation`

DreamWaQ算法相关入口位于：

- [`rsl_rl/algorithms/dwaq_ppo.py`](./rsl_rl/algorithms/dwaq_ppo.py)
- [`rsl_rl/modules/actor_critic_DWAQ.py`](./rsl_rl/modules/actor_critic_DWAQ.py)
- [`rsl_rl/runners/on_policy_runner.py`](./rsl_rl/runners/on_policy_runner.py)

## 安装

建议使用 editable 安装：

```bash
git clone <your-repo-url>
cd rsl_rl
pip install -e .
```
（只是调一下包的话，下面的部分其实并没有必要多做研究，直接把代码丢给codex，它可以很好的帮你配置好你的Isaaclab调用）

## 调用方式

标准 `rsl_rl` 的核心调用链是：

1. 环境返回 `TensorDict`
2. `OnPolicyRunner` 读取 `obs_groups`
3. policy 从 `TensorDict` 中提取 actor / critic 所需观测
4. algorithm 调用 `act / process_env_step / compute_returns / update`

这也是当前 DreamWaQ算法 的调用方式。也就是说，DreamWaQ算法 不再需要一条单独的训练分支。

## DreamWaQ算法 接入说明

### 1. 观测协议

如果要使用DreamWaQ算法，环境返回的 `TensorDict` 至少需要包含这些 key：

- `policy`
- `obs_history` 或 `obs_hist`
- `privileged` 或其他供 critic 使用的 group

最常见的观测组织方式如下：

```python
from tensordict import TensorDict

obs = TensorDict(
    {
        "policy": actor_obs,          # [num_envs, actor_dim]
        "privileged": privileged_obs, # [num_envs, privileged_dim]
        "obs_history": obs_history,   # [num_envs, actor_dim * history_len]
    },
    batch_size=[num_envs],
)
```

其中：

- `policy` 是 actor 当前时刻观测
- `obs_history` 是展平后的历史观测
- critic 的输入由 `obs_groups["critic"]` 决定

### 2. `obs_groups` 约定

DreamWaQ算法仍然沿用原生 `obs_groups`：

```yaml
obs_groups: {"policy": ["policy"], "critic": ["policy", "privileged"]}
```

这表示：

- actor 只吃 `policy`
- critic 吃 `policy + privileged`

### 3. 重要约束

DreamWaQ算法当前的速度监督仍采用如下切片逻辑：

```python
vel_target = critic_obs[:, obs_dim : obs_dim + 3]
```

这意味着 critic 输入的拼接顺序需要满足：

1. 前 `obs_dim` 维对应 actor 观测
2. 紧接着 3 维是目标速度

如果你的 critic 观测顺序不是这个结构，需要同步修改 DreamWaQ算法 loss。

## DreamWaQ算法 配置指南

下面是一份最小可用配置片段，调用侧风格与原生 PPO 基本一致。

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

项目仍然保留了 `DWAQOnPolicyRunner` 这个类名，也可以继续使用；它当前只是兼容包装，推荐优先用 `OnPolicyRunner`。

## DreamWaQ算法 最小环境示例

下面给出一个最小环境侧返回格式示例：

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

## 训练侧最小示例

如果你的外部仓库是通过已安装的 `rsl_rl` 调用训练器，最小形态如下：

```python
from rsl_rl.runners import OnPolicyRunner

env = MyEnv()
runner_cfg = train_cfg["runner"]
runner = OnPolicyRunner(env, runner_cfg, log_dir="logs/demo", device="cuda:0")
runner.learn(num_learning_iterations=runner_cfg["max_iterations"])
```

## 致谢

原始 `rsl_rl` 由 ETH Zurich Robotic Systems Lab 与 NVIDIA 维护。贡献者请参考：

- [CONTRIBUTORS.md](./CONTRIBUTORS.md)
- [CITATION.cff](./CITATION.cff)
