import argparse
import json
import warnings
from pathlib import Path

import gymnasium as gym
import gymnasium_robotics
import imageio.v2 as imageio
from stable_baselines3 import SAC


DEFAULT_ENV_ID = "FetchReach-v4"
SCRIPT_DIR = Path(__file__).resolve().parent
REPO_ROOT = SCRIPT_DIR.parents[1]


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


def resolve_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    cwd_path = Path.cwd() / path
    if cwd_path.exists():
        return cwd_path
    return REPO_ROOT / path


def resolve_output_path(value: str) -> Path:
    path = Path(value)
    if path.is_absolute():
        return path
    return REPO_ROOT / path


def main() -> None:
    parser = argparse.ArgumentParser(description="Render trained SAC+HER policy on a Gymnasium-Robotics Fetch task.")
    parser.add_argument("--env-id", default=DEFAULT_ENV_ID)
    parser.add_argument("--task-name", default=None)
    parser.add_argument(
        "--model",
        default="models/fetch-reach/sac_her_fetch_reach_final.zip",
    )
    parser.add_argument("--out", default="results/fetch-reach")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--fps", type=int, default=8)
    parser.add_argument("--max-steps", type=int, default=50)
    parser.add_argument("--gif-name", default="demo.gif")
    parser.add_argument("--mp4-name", default="demo.mp4")
    args = parser.parse_args()

    warnings.filterwarnings("ignore", message=".*AdroitHand.*")

    task_name = args.task_name or task_name_from_env_id(args.env_id)
    model_path = resolve_path(args.model)
    out_dir = resolve_output_path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    gif_path = out_dir / args.gif_name
    mp4_path = out_dir / args.mp4_name
    summary_path = out_dir / "render_summary.json"

    gym.register_envs(gymnasium_robotics)
    env = gym.make(args.env_id, render_mode="rgb_array")
    model = SAC.load(model_path, env=env)

    obs, _ = env.reset(seed=args.seed)
    frames = []
    total_reward = 0.0
    final_success = 0.0
    steps = 0

    for _ in range(args.max_steps):
        frame = env.render()
        if frame is not None:
            frames.append(frame)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        total_reward += float(reward)
        final_success = float(info.get("is_success", 0.0))
        steps += 1

        if terminated or truncated:
            frame = env.render()
            if frame is not None:
                frames.append(frame)
            break

    env.close()

    if not frames:
        raise RuntimeError("No frames were rendered. Check MuJoCo/OpenGL render support.")

    imageio.mimsave(gif_path, frames, fps=args.fps)
    mp4_error = None
    try:
        imageio.mimsave(mp4_path, frames, fps=args.fps)
    except ValueError as error:
        mp4_error = str(error)

    summary = {
        "env_id": args.env_id,
        "task_name": task_name,
        "model": str(model_path),
        "seed": args.seed,
        "steps": steps,
        "total_reward": total_reward,
        "final_success": final_success,
        "frame_count": len(frames),
        "fps": args.fps,
        "gif_path": str(gif_path),
        "mp4_path": str(mp4_path) if mp4_error is None else None,
        "mp4_error": mp4_error,
    }
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
