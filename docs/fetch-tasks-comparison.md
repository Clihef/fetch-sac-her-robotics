# Fetch 系列任务 SAC + HER 复现实验对比

本文档记录基于 Gymnasium-Robotics 和 MuJoCo 的 Fetch 系列机械臂任务复现实验。所有任务使用 Stable-Baselines3 的 SAC + HER 训练，策略为 `MultiInputPolicy`，replay buffer 为 `HerReplayBuffer`。

## 任务设置

| 任务 | 环境 ID | 目标 |
| --- | --- | --- |
| Reach | `FetchReach-v4` | 控制机械臂末端执行器到达目标点 |
| Push | `FetchPush-v4` | 推动物体到达桌面目标位置 |
| Slide | `FetchSlide-v4` | 将物体滑动到远处目标区域 |
| PickAndPlace | `FetchPickAndPlace-v4` | 抓取物体并移动到目标位置 |

四个任务均为 GoalEnv，观测包含：

- `observation`: 机器人和物体状态；
- `achieved_goal`: 当前实际达到的目标状态；
- `desired_goal`: 期望目标状态。

奖励使用 sparse reward：未达到目标为 `-1`，达到目标为 `0`。HER 通过目标重标记增强失败轨迹的样本利用率。

## 实验结果

| Task | Output Dir | Budget Steps | Effective Final Steps | Final Success Rate | Best Success Rate | Mean Reward at Final Eval | GIF |
| --- | --- | ---: | ---: | ---: | ---: | ---: | --- |
| FetchReach-v4 | `results/fetch-reach` | 10k | 10k | 1.00 | 1.00 | -1.60 | `results/fetch-reach/demo.gif` |
| FetchPush-v4 | `results/fetch-push` | 1M | 600k | 0.90 | 0.90 | -11.90 | `results/fetch-push/demo.gif` |
| FetchSlide-v4 | `results/fetch-slide` | 1.2M | 1.2M | 0.70 | 0.85 | -30.35 | `results/fetch-slide/demo.gif` |
| FetchPickAndPlace-v4 | `results/fetch-pick-and-place` | 800k | 620k | 0.95 | 1.00 | -12.85 | `results/fetch-pick-and-place/demo.gif` |

## 结果解读

`FetchReach-v4` 是最简单的基线任务，不涉及物体接触和抓取，仅需要控制末端执行器接近目标点，因此在 10k steps 内就能达到较高成功率。

`FetchPush-v4` 引入了物体和桌面接触，策略需要先移动到物体附近，再通过接触推动物体。该任务在继续训练后达到 0.90 success rate，并触发早停。

`FetchPickAndPlace-v4` 同时涉及接近、抓取、抬升和移动物体，任务链路更长。继续训练后 final success rate 达到 0.95，说明 SAC + HER 能够在该目标条件任务中形成有效策略。

`FetchSlide-v4` 难度主要来自滑动物体的惯性和接触动力学。策略不能简单地把物体推到目标点，而需要学习合适的击打方向和力度。长训练后最高 success rate 达到 0.85，但最终评估为 0.70，仍存在明显波动，因此不应表述为稳定收敛。

## 面试表述建议

可以这样描述项目：

> 我基于 Gymnasium-Robotics 和 MuJoCo 复现了 Fetch 系列机械臂目标条件任务，使用 Stable-Baselines3 的 SAC + HER 搭建训练和评估流程。Fetch 任务返回 `observation / achieved_goal / desired_goal` 字典观测，因此使用 `MultiInputPolicy` 处理多输入状态；由于 sparse reward 下失败样本很多，我使用 HER 对失败轨迹进行目标重标记，并通过环境的 `compute_reward` 重新计算奖励。实验覆盖 Reach、Push、Slide、PickAndPlace 四类任务，Push 和 PickAndPlace 达到 0.90 以上成功率，Slide 由于滑动接触动力学更复杂，表现出更明显的训练波动。

需要避免夸大：

- 不说真实机器人部署；
- 不说熟练掌握 ROS 或 Isaac；
- 不说 Slide 已稳定完全收敛；
- 不把 Gymnasium-Robotics 说成自己开发的机器人系统。
