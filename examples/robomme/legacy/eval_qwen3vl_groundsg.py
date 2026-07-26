"""RoboMME evaluation: base Qwen3-VL object memory planner + GroundSG pi0.5."""

# ruff: noqa: SLF001

from __future__ import annotations

import dataclasses
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from legacy import eval_gpt5nano_groundsg as shared_eval
from openpi_client import websocket_client_policy
from subgoal_prediction.legacy.gpt5nano_groundsg.schemas import FailureAudit
from subgoal_prediction.legacy.qwen3vl_groundsg import MODEL
from subgoal_prediction.legacy.qwen3vl_groundsg import Qwen3VLGroundSGPlanner
from subgoal_prediction.legacy.qwen3vl_groundsg.online_planner import DEFAULT_MODEL_DIR

POLICY_NAME = "ours_ablation"
PLANNER_INTERVAL = 16


@dataclasses.dataclass
class Args(shared_eval.Args):
    planner_model_dir: str = str(DEFAULT_MODEL_DIR)
    planner_max_retries: int = 3
    planner_max_tokens: int = 4096


def _output_root(args: Args) -> Path:
    return Path(args.save_dir) / POLICY_NAME / f"ckpt{args.model_ckpt_id}" / f"seed{args.model_seed}" / MODEL


def _checkpoint_files(model_dir: Path) -> tuple[list[str], list[str]]:
    index_path = model_dir / "model.safetensors.index.json"
    if not index_path.is_file():
        return [], ["model.safetensors.index.json"]
    index = json.loads(index_path.read_text(encoding="utf-8"))
    shards = sorted(set(index.get("weight_map", {}).values()))
    required = [
        "config.json",
        "model.safetensors.index.json",
        "preprocessor_config.json",
        "tokenizer.json",
        *shards,
    ]
    return shards, [name for name in required if not (model_dir / name).is_file()]


def validate_setup(args: Args) -> dict[str, Any]:
    root = Path(__file__).resolve().parents[3]
    package = (
        Path(__file__).resolve().parents[1]
        / "subgoal_prediction"
        / "legacy"
        / "gpt5nano_groundsg"
    )
    task_files = sorted((package / "task_contexts").glob("*.txt"))
    missing_contexts = sorted(set(shared_eval.TASK_NAME_LIST) - {path.stem for path in task_files})
    model_dir = (root / args.planner_model_dir).resolve()
    shards, missing_model_files = _checkpoint_files(model_dir)
    policy_checkpoint = root / "runs" / "ckpts" / "symbolic-grounded-subgoal" / str(args.model_ckpt_id)
    checks = {
        "model": MODEL,
        "model_dir": str(model_dir),
        "base_instruct_only": True,
        "adapter": None,
        "model_shards": shards,
        "missing_model_files": missing_model_files,
        "planner_interval": PLANNER_INTERVAL,
        "task_context_count": len(task_files),
        "missing_task_contexts": missing_contexts,
        "policy_checkpoint": str(policy_checkpoint),
        "policy_checkpoint_exists": policy_checkpoint.is_dir(),
        "output_root": str(_output_root(args)),
        "selected_tasks": shared_eval._selected_tasks(args),
        "episodes_per_task": args.num_episodes,
        "cuda_visible_devices": os.getenv("CUDA_VISIBLE_DEVICES"),
    }
    if missing_contexts:
        raise RuntimeError(f"Missing task contexts: {missing_contexts}")
    if missing_model_files:
        raise RuntimeError(f"Qwen checkpoint is incomplete; missing files: {missing_model_files}")
    if len(shards) != 2:
        raise RuntimeError(f"Expected exactly two Qwen safetensors shards, found {shards}")
    if not policy_checkpoint.is_dir():
        raise RuntimeError(f"GroundSG policy checkpoint not found: {policy_checkpoint}")
    if args.num_episodes != 2:
        print(f"[qwen] Warning: requested {args.num_episodes} episodes instead of 2.")
    return checks


class EpisodeEvaluator(shared_eval.EpisodeEvaluator):
    def __init__(self, args: Args, output_root: Path) -> None:
        self.args = args
        self.output_root = output_root
        self.client = websocket_client_policy.MMEVLAWebsocketClientPolicy(
            args.host,
            args.port,
        )
        self.planner = Qwen3VLGroundSGPlanner(
            output_root / "planner_logs",
            model_dir=args.planner_model_dir,
            max_retries=args.planner_max_retries,
            max_tokens=args.planner_max_tokens,
        )


def evaluate(args: Args) -> None:
    checks = validate_setup(args)
    print(json.dumps(checks, indent=2))
    if args.dry_run:
        print("[qwen] Dry run passed; no server connection or model load was made.")
        return

    output_root = _output_root(args)
    if output_root.exists() and args.overwrite:
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    video_directory = output_root / "videos"
    video_directory.mkdir(parents=True, exist_ok=True)

    progress_path = output_root / "progress.json"
    episode_results_path = output_root / "episode_results.json"
    progress = json.loads(progress_path.read_text()) if progress_path.is_file() else {}
    episode_results = json.loads(episode_results_path.read_text()) if episode_results_path.is_file() else {}
    failure_csv = output_root / "failure_episodes.csv"
    report_jsonl = output_root / "task_reports.jsonl"
    re_eval = {task.strip() for task in args.re_eval_tasks.split(",") if task.strip()}
    for task in re_eval:
        progress.pop(task, None)
        episode_results.pop(task, None)

    evaluator = EpisodeEvaluator(args, output_root)
    try:
        for task in shared_eval._selected_tasks(args):
            progress.setdefault(task, {})
            episode_results.setdefault(task, {})
            env_runner = shared_eval.EnvRunner(task, video_directory, max_steps=args.max_steps)
            try:
                count = min(args.num_episodes, env_runner.num_episodes)
                for episode in range(count):
                    key = str(episode)
                    previous = episode_results[task].get(key)
                    if previous and previous.get("outcome") != "error":
                        print(f"[qwen] {task} ep{episode} already complete; skipping.")
                        continue
                    env_runner.make_env(episode)
                    print(f"\n[qwen] Starting {task} ep{episode} with {MODEL}")
                    try:
                        outcome, details, audit = evaluator.run_episode(
                            env_runner,
                            video_directory,
                        )
                    except Exception as error:
                        outcome = "error"
                        details = {
                            "task": task,
                            "episode": episode,
                            "outcome": outcome,
                            "success": False,
                            "error": repr(error),
                        }
                        audit = FailureAudit(
                            first_failure_step=None,
                            primary_failure="environment_error",
                            secondary_failure=None,
                            confidence="low",
                            notes=f"Evaluation error: {error}",
                        )
                        details["failure_audit"] = audit.model_dump(mode="json")
                        print(f"[qwen] ERROR {task} ep{episode}: {error}")
                    finally:
                        env_runner.close_env()

                    progress[task][key] = outcome == "success"
                    episode_results[task][key] = details
                    shared_eval._save_json(progress_path, progress)
                    shared_eval._save_json(episode_results_path, episode_results)
                    shared_eval._update_failure_csv(
                        failure_csv,
                        task,
                        episode,
                        outcome,
                        audit if outcome != "success" else None,
                    )
                    print(
                        f"[qwen] Finished {task} ep{episode}: {outcome}; "
                        f"failure={audit.primary_failure if audit else 'none'}"
                    )
            finally:
                env_runner.close_env()

            report = shared_eval._task_report(task, episode_results)
            with report_jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
            shared_eval._save_json(
                output_root / "task_summaries" / f"{task}.json",
                report,
            )
            print(
                f"\n[qwen] TASK COMPLETE {task}: "
                f"{report['success']}/{report['episodes']} success "
                f"({report['success_rate']:.1%})"
            )
            time.sleep(1)

        reports = {
            task: shared_eval._task_report(task, episode_results)
            for task in shared_eval._selected_tasks(args)
            if task in episode_results
        }
        success_rates = {task: report["success_rate"] for task, report in reports.items()}
        values = list(success_rates.values())
        shared_eval._save_json(
            output_root / "log.json",
            {
                "model": MODEL,
                "model_dir": str(Path(args.planner_model_dir).resolve()),
                "adapter": None,
                "planner_interval": PLANNER_INTERVAL,
                "success_rate": success_rates,
                "total_success_rate": sum(values) / len(values) if values else 0.0,
            },
        )
    finally:
        evaluator.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(evaluate)
