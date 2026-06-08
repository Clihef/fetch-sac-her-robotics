# SAC + HER 实验结果记录

本文档记录 Fetch 系列机械臂任务的主要训练结果。项目使用 Gymnasium-Robotics 提供任务环境，MuJoCo 负责物理仿真，Stable-Baselines3 实现 SAC + HER。

## 训练配置

- 算法：SAC + HER
- Policy：`MultiInputPolicy`
- Replay Buffer：`HerReplayBuffer`
- HER goal selection：`future`
- HER sampled goals：`n_sampled_goal = 4`
- 奖励形式：sparse reward
- 评估方式：定期 evaluation，记录 mean reward、success rate、episode length
- 模型文件：不随仓库上传，结果目录保留 CSV、曲线、summary 和 GIF

## 结果汇总

| Task | Env ID | Final Steps | Final Success Rate | Best Success Rate | Final Mean Reward |
| --- | --- | ---: | ---: | ---: | ---: |
| Reach | `FetchReach-v4` | 10k | 1.00 | 1.00 | -1.60 |
| Push | `FetchPush-v4` | 600k | 0.90 | 0.90 | -11.90 |
| Slide | `FetchSlide-v4` | 1.2M | 0.70 | 0.85 | -30.35 |
| PickAndPlace | `FetchPickAndPlace-v4` | 620k | 0.95 | 1.00 | -12.85 |

## 输出文件

每个任务的公开产物位于 `results/`：

```text
results/<task>/
├── eval_metrics.csv
├── eval_curve.png
├── summary.json
├── render_summary.json
└── demo.gif
```

其中：

- `eval_metrics.csv` 记录每次 evaluation 的数值；
- `eval_curve.png` 展示 mean reward 和 success rate 曲线；
- `summary.json` 记录训练配置、最后评估结果和是否早停；
- `render_summary.json` 记录 GIF 渲染配置和最终可视化结果；
- `demo.gif` 展示训练后策略在 MuJoCo 中的执行效果。

## 任务难度分析

`FetchReach-v4` 不涉及物体接触，是最简单的末端到达任务，因此 10k steps 内即可达到高成功率。

`FetchPush-v4` 需要通过机械臂与物体接触，将物体推到目标位置。它比 Reach 更难，但在继续训练后达到 0.90 success rate。

`FetchPickAndPlace-v4` 包含接近、抓取、抬升和移动多个阶段。继续训练后达到 0.95 final success rate，说明 HER 对这类目标条件任务有效。

`FetchSlide-v4` 的难点在于滑动物体的惯性和接触动力学。策略需要学习合适的击打方向和力度。实验中最高 success rate 达到 0.85，但最终为 0.70，仍有波动，因此只表述为“有明显提升但未稳定收敛”。

## 渲染命令示例

如果本地有训练好的模型，可使用如下命令生成 GIF：

```powershell
& .\.venv\Scripts\python.exe scripts\render_fetch_policy.py `
  --env-id FetchPickAndPlace-v4 `
  --task-name fetch-pick-and-place `
  --model models\fetch-pick-and-place\sac_her_fetch_pick_and_place_final.zip `
  --out results\fetch-pick-and-place `
  --gif-name demo.gif `
  --fps 4 `
  --max-steps 50 `
  --seed 42
```

## 简历表述

基于 Gymnasium-Robotics 和 MuJoCo 复现 Fetch 系列机械臂任务，使用 SAC + HER 搭建 sparse reward 下的目标条件强化学习训练流程；通过 `MultiInputPolicy` 处理 `observation / achieved_goal / desired_goal` 字典观测，利用 HER 进行目标重标记，并输出评估曲线、success rate 和 MuJoCo GIF 可视化结果。实验覆盖 Reach、Push、Slide、PickAndPlace 四类任务，其中 Push 和 PickAndPlace 达到 0.90 以上成功率，Slide 任务表现出更强的接触动力学难度和训练波动。
