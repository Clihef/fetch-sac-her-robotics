# Fetch SAC + HER Robotics Reproduction

本项目基于 Gymnasium-Robotics 和 MuJoCo 复现 Fetch 系列机械臂目标条件控制任务，并使用 Stable-Baselines3 的 SAC + HER 搭建训练、评估和可视化流程。

项目重点不是实机部署，而是完成一个可复现、可解释、可展示的机器人强化学习仿真实验闭环：环境创建、GoalEnv 字典观测处理、HER 目标重标记、SAC 连续控制训练、评估曲线记录和 GIF 可视化。

## Tasks

支持以下 Fetch 系列任务：

| Environment | Task | Description |
| --- | --- | --- |
| `FetchReach-v4` | Reach | 控制机械臂末端到达目标点 |
| `FetchPush-v4` | Push | 推动物体到目标位置 |
| `FetchSlide-v4` | Slide | 滑动物体到远处目标位置 |
| `FetchPickAndPlace-v4` | PickAndPlace | 抓取物体并移动到目标位置 |

## Method

训练算法使用：

- SAC: off-policy 连续控制 actor-critic 算法
- HER: Hindsight Experience Replay，用于 sparse reward 的目标重标记
- `MultiInputPolicy`: 处理 GoalEnv 返回的字典观测
- `HerReplayBuffer`: 对 `achieved_goal` 和 `desired_goal` 进行 goal relabeling

Fetch 任务的 observation 是字典结构：

```python
{
    "observation": robot_and_object_state,
    "achieved_goal": current_goal_state,
    "desired_goal": target_goal_state,
}
```

HER 会把失败轨迹中实际达到的状态重新标记为目标，并调用环境的 `compute_reward` 重新计算奖励，使稀疏奖励下的失败经验也能用于训练。

## Repository Layout

```text
.
├── scripts/
│   ├── train_fetch_sac_her.py
│   └── render_fetch_policy.py
├── docs/
│   ├── NOTEME.md
│   ├── fetch-tasks-comparison.md
│   ├── fetch-task-source-walkthrough.md
│   └── sac-her-results.md
├── results/
│   ├── fetch-reach/
│   ├── fetch-push/
│   ├── fetch-slide/
│   └── fetch-pick-and-place/
├── models/
│   └── README.md
├── requirements.txt
└── .gitignore
```

## Installation

建议使用 Python 3.10 或 3.11。

```bash
python -m venv .venv
.\.venv\Scripts\activate
pip install -r requirements.txt
```

如果使用本地 editable 版本的 Gymnasium-Robotics，可执行：

```bash
pip install -e path/to/Gymnasium-Robotics
```

## Training

Reach 任务 10k smoke run：

```bash
python scripts/train_fetch_sac_her.py ^
  --env-id FetchReach-v4 ^
  --task-name fetch-reach ^
  --out results/fetch-reach ^
  --total-timesteps 10000 ^
  --eval-freq 2000 ^
  --eval-episodes 5 ^
  --seed 42
```

Push 任务长训练示例：

```bash
python scripts/train_fetch_sac_her.py ^
  --env-id FetchPush-v4 ^
  --task-name fetch-push ^
  --out results/fetch-push ^
  --total-timesteps 1000000 ^
  --eval-freq 20000 ^
  --eval-episodes 20 ^
  --seed 43 ^
  --target-success-rate 0.9 ^
  --success-patience 2
```

Slide 和 PickAndPlace 可替换 `--env-id` 与 `--task-name` 运行：

```bash
python scripts/train_fetch_sac_her.py --env-id FetchSlide-v4 --task-name fetch-slide --out results/fetch-slide
python scripts/train_fetch_sac_her.py --env-id FetchPickAndPlace-v4 --task-name fetch-pick-and-place --out results/fetch-pick-and-place
```

## Rendering

如果本地已有模型文件，可生成 GIF：

```bash
python scripts/render_fetch_policy.py ^
  --env-id FetchPickAndPlace-v4 ^
  --task-name fetch-pick-and-place ^
  --model path/to/sac_her_fetch_pick_and_place_final.zip ^
  --out results/fetch-pick-and-place ^
  --gif-name demo.gif ^
  --fps 4 ^
  --max-steps 50 ^
  --seed 42
```

本仓库默认不上传模型 zip 文件。`results/` 中保留了评估曲线、CSV、summary 和 GIF，用于展示实验结果。

## Results

| Task | Train Steps | Final Success Rate | Best Success Rate | Notes |
| --- | ---: | ---: | ---: | --- |
| FetchReach-v4 | 10k | 1.00 | 1.00 | 简单 baseline，10k 内即可达到较高成功率 |
| FetchPush-v4 | 600k effective / 1M budget | 0.90 | 0.90 | 达到 0.90 早停阈值 |
| FetchPickAndPlace-v4 | 620k effective / 800k budget | 0.95 | 1.00 | 达到 0.90 早停阈值 |
| FetchSlide-v4 | 1.2M | 0.70 | 0.85 | 有提升但仍不稳定，不包装为完全收敛 |

GIF 示例位于：

```text
results/fetch-reach/demo.gif
results/fetch-push/demo.gif
results/fetch-slide/demo.gif
results/fetch-pick-and-place/demo.gif
```

## Project Boundary

本项目是本地仿真复现项目：

- 使用 MuJoCo 进行物理仿真
- 使用 Gymnasium-Robotics 提供 Fetch 任务环境
- 使用 Stable-Baselines3 实现 SAC + HER
- 不涉及真实机器人部署
- 不涉及 ROS 控制
- 不涉及 Isaac Sim

