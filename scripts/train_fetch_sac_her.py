import argparse
import csv
import json
import time
import warnings
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import matplotlib.pyplot as plt
import numpy as np
from stable_baselines3 import SAC
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.her.her_replay_buffer import HerReplayBuffer


DEFAULT_ENV_ID = "FetchReach-v4"


def task_name_from_env_id(env_id: str) -> str:
    text = env_id
    for suffix in ("Dense-v4", "Dense-v1", "-v4", "-v1"):
        text = text.replace(suffix, "")
    parts = []
    current = ""
    for char in text:
        if char.isupper() and current:
            parts.append(current.lower())
            current = char
        else:
            current += char
    if current:
        parts.append(current.lower())
    return "-".join(parts)


def make_env(seed: int, env_id: str):
    gym.register_envs(gymnasium_robotics)
    env = gym.make(env_id)
    env.reset(seed=seed)
    return Monitor(env)


def evaluate(model: SAC, seed: int, episodes: int, env_id: str) -> dict:
    env = make_env(seed, env_id)
    episode_rewards = []
    episode_lengths = []
    successes = []

    for episode in range(episodes):
        obs, _ = env.reset(seed=seed + episode)
        done = False
        episode_reward = 0.0
        episode_length = 0
        final_success = 0.0

        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += float(reward)
            episode_length += 1
            final_success = float(info.get("is_success", 0.0))
            done = bool(terminated or truncated)

        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        successes.append(final_success)

    env.close()
    return {
        "mean_reward": float(np.mean(episode_rewards)),
        "success_rate": float(np.mean(successes)),
        "mean_episode_length": float(np.mean(episode_lengths)),
        "episodes": episodes,
    }


def write_eval_row(path: Path, row: dict) -> None:
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "total_timesteps",
                "mean_reward",
                "success_rate",
                "mean_episode_length",
                "episodes",
                "elapsed_seconds",
            ],
        )
        if not exists:
            writer.writeheader()
        writer.writerow(row)


def plot_metrics(csv_path: Path, png_path: Path, title: str) -> None:
    timesteps = []
    rewards = []
    successes = []

    with csv_path.open("r", encoding="utf-8") as file:
        reader = csv.DictReader(file)
        for row in reader:
            timesteps.append(int(row["total_timesteps"]))
            rewards.append(float(row["mean_reward"]))
            successes.append(float(row["success_rate"]))

    if not timesteps:
        return

    fig, reward_ax = plt.subplots(figsize=(8, 4.5), dpi=140)
    success_ax = reward_ax.twinx()

    reward_ax.plot(timesteps, rewards, marker="o", color="#0f766e", label="mean reward")
    success_ax.plot(timesteps, successes, marker="s", color="#b45309", label="success rate")

    reward_ax.set_xlabel("total timesteps")
    reward_ax.set_ylabel("mean reward")
    success_ax.set_ylabel("success rate")
    reward_ax.grid(True, alpha=0.25)
    reward_ax.set_title(title)

    lines, labels = reward_ax.get_legend_handles_labels()
    lines2, labels2 = success_ax.get_legend_handles_labels()
    reward_ax.legend(lines + lines2, labels + labels2, loc="best")

    fig.tight_layout()
    fig.savefig(png_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train SAC + HER on a Gymnasium-Robotics Fetch task.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--task-name", default=None)
    parser.add_argument("--model-prefix", default=None)
    parser.add_argument("--resume-from", default=None)
    parser.add_argument("--initial-timesteps", type=int, default=0)
    parser.add_argument("--resume-warmup-steps", type=int, default=500)
    parser.add_argument("--out", default="results/fetch-reach")
    parser.add_argument("--total-timesteps", type=int, default=10000)
    parser.add_argument("--eval-freq", type=int, default=2000)
    parser.add_argument("--eval-episodes", type=int, default=5)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--verbose", type=int, default=1)
    parser.add_argument(
        "--target-success-rate",
        type=float,
        default=None,
        help="Stop early after success rate reaches this value for enough consecutive evaluations.",
    )
    parser.add_argument(
        "--success-patience",
        type=int,
        default=2,
        help="Number of consecutive evaluations required for early stopping.",
    )
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*AdroitHand.*")

    task_name = args.task_name or task_name_from_env_id(args.env_id)
    model_prefix = args.model_prefix or f"sac_her_{task_name.replace('-', '_')}_final"

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_path = out_dir / "eval_metrics.csv"
    png_path = out_dir / "eval_curve.png"
    summary_path = out_dir / "summary.json"
    model_path = out_dir / model_prefix
    checkpoint_path = out_dir / f"{model_prefix}_latest"

    env = make_env(args.seed, args.env_id)
    if args.resume_from:
        model = SAC.load(args.resume_from, env=env, verbose=args.verbose)
        model.learning_starts = max(
            int(model.num_timesteps) + max(0, args.resume_warmup_steps),
            int(getattr(model, "learning_starts", 0)),
        )
    else:
        model = SAC(
            "MultiInputPolicy",
            env,
            replay_buffer_class=HerReplayBuffer,
            replay_buffer_kwargs={
                "n_sampled_goal": 4,
                "goal_selection_strategy": "future",
            },
            learning_starts=500,
            buffer_size=100_000,
            batch_size=256,
            gamma=0.95,
            tau=0.05,
            learning_rate=1e-3,
            train_freq=1,
            gradient_steps=1,
            ent_coef="auto",
            verbose=args.verbose,
            seed=args.seed,
        )

    start = time.time()
    completed = args.initial_timesteps
    eval_freq = max(1, args.eval_freq)
    consecutive_successes = 0
    stopped_early = False
    stop_reason = None
    last_metrics = None

    while completed < args.total_timesteps:
        chunk = min(eval_freq, args.total_timesteps - completed)
        model.learn(total_timesteps=chunk, reset_num_timesteps=False, progress_bar=False)
        completed += chunk

        metrics = evaluate(model, args.seed + 10_000 + completed, args.eval_episodes, args.env_id)
        metrics["total_timesteps"] = completed
        metrics["elapsed_seconds"] = round(time.time() - start, 2)
        write_eval_row(csv_path, metrics)
        model.save(checkpoint_path)
        print(json.dumps(metrics, ensure_ascii=False))
        last_metrics = metrics

        if args.target_success_rate is not None:
            if metrics["success_rate"] >= args.target_success_rate:
                consecutive_successes += 1
            else:
                consecutive_successes = 0

            if consecutive_successes >= max(1, args.success_patience):
                stopped_early = True
                stop_reason = (
                    f"success_rate >= {args.target_success_rate} "
                    f"for {consecutive_successes} consecutive evaluations"
                )
                break

    model.save(model_path)
    env.close()
    plot_metrics(csv_path, png_path, f"{args.env_id} SAC + HER Evaluation")

    summary = {
        "env_id": args.env_id,
        "task_name": task_name,
        "algorithm": "SAC + HER",
        "policy": "MultiInputPolicy",
        "total_timesteps": args.total_timesteps,
        "eval_freq": args.eval_freq,
        "eval_episodes": args.eval_episodes,
        "seed": args.seed,
        "resume_from": args.resume_from,
        "initial_timesteps": args.initial_timesteps,
        "resume_warmup_steps": args.resume_warmup_steps,
        "target_success_rate": args.target_success_rate,
        "success_patience": args.success_patience,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
        "model_path": str(model_path.with_suffix(".zip")),
        "checkpoint_model_path": str(checkpoint_path.with_suffix(".zip")),
        "eval_metrics_csv": str(csv_path),
        "eval_curve_png": str(png_path),
        "last_metrics": last_metrics,
        "elapsed_seconds": round(time.time() - start, 2),
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")


if __name__ == "__main__":
    main()
