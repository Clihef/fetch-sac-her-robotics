# Models

模型 zip 文件默认不纳入仓库，原因是：

- 文件体积相对较大；
- 仓库主要用于展示训练代码、实验记录、曲线和 GIF；
- 可根据 `scripts/train_fetch_sac_her.py` 和各任务 `summary.json` 中的配置重新训练。

如需复现实验，可按 README 中的训练命令重新生成模型，再使用 `scripts/render_fetch_policy.py` 生成 GIF。
