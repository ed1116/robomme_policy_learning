"""Full RoboMME evaluation: GPT-5 Nano object memory + GroundSG pi0.5.

This is a separate evaluation entrypoint. Existing GroundSG/QwenVL files are
not modified.
"""

from __future__ import annotations

import csv
import dataclasses
import json
import os
from pathlib import Path
import shutil
import time
from typing import Any

from env_runner import EnvRunner
import numpy as np
from openpi_client import websocket_client_policy
from subgoal_prediction.gpt5nano_groundsg import GPT5NanoGroundSGPlanner
from subgoal_prediction.gpt5nano_groundsg.online_planner import FRAME_COUNT
from subgoal_prediction.gpt5nano_groundsg.online_planner import MODEL
from subgoal_prediction.gpt5nano_groundsg.online_planner import _sample_four
from subgoal_prediction.gpt5nano_groundsg.schemas import FailureAudit
from utils import TASK_NAME_LIST
from utils import TASK_WITH_VIDEO_DEMO
from utils import EpisodeState
from utils import RolloutRecorder
from utils import pack_buffer

POLICY_NAME = "ours_ablation"
PLANNER_INTERVAL = 16
FAILURE_COLUMNS = [
    "suite",
    "task",
    "episode(0~9)",
    "outcome",
    "first_failure_step",
    "primary_failure",
    "secondary_failure",
    "confidence",
    "notes",
]


@dataclasses.dataclass
class Args:
    host: str = "127.0.0.1"
    port: int = 8011
    max_steps: int = 1300
    num_episodes: int = 2
    model_seed: int = 7
    model_ckpt_id: int = 79999
    save_dir: str = "runs/evaluation"
    only_tasks: str = ""
    exclude_tasks: str = ""
    re_eval_tasks: str = ""
    overwrite: bool = False
    audit_failures: bool = True
    shutdown_server: bool = True
    dry_run: bool = False


def _jsonable(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, bytes):
        return value.decode("utf-8", errors="replace")
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, list | tuple):
        return [_jsonable(v) for v in value]
    return value


def _diagnostic_snapshot(info: dict[str, Any], step: int) -> dict[str, Any]:
    result: dict[str, Any] = {"step": step}
    for key in (
        "simple_subgoal_online",
        "grounded_subgoal_online",
        "simple_subgoal",
        "grounded_subgoal",
        "is_completed",
        "status",
    ):
        if key in info:
            result[key] = _jsonable(info[key])
    return result


def _append_changed(trace: list[dict[str, Any]], item: dict[str, Any]) -> None:
    comparable = {k: v for k, v in item.items() if k != "step"}
    if trace and {k: v for k, v in trace[-1].items() if k != "step"} == comparable:
        return
    trace.append(item)


def _save_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")
    temporary.replace(path)


def _output_root(args: Args) -> Path:
    return Path(args.save_dir) / POLICY_NAME / f"ckpt{args.model_ckpt_id}" / f"seed{args.model_seed}" / MODEL


def _selected_tasks(args: Args) -> list[str]:
    tasks = args.only_tasks.split(",") if args.only_tasks else list(TASK_NAME_LIST)
    tasks = [task.strip() for task in tasks if task.strip()]
    excluded = {task.strip() for task in args.exclude_tasks.split(",") if task.strip()}
    tasks = [task for task in tasks if task not in excluded]
    invalid = sorted(set(tasks) - set(TASK_NAME_LIST))
    if invalid:
        raise RuntimeError(f"Unknown tasks: {invalid}")
    return tasks


def validate_setup(args: Args) -> dict[str, Any]:
    package_root = Path(__file__).resolve().parent / "subgoal_prediction" / "gpt5nano_groundsg"
    task_files = sorted((package_root / "task_contexts").glob("*.txt"))
    missing = sorted(set(TASK_NAME_LIST) - {path.stem for path in task_files})
    checkpoint = Path("runs/ckpts/symbolic-grounded-subgoal") / str(args.model_ckpt_id)
    checks = {
        "model": MODEL,
        "planner_interval": PLANNER_INTERVAL,
        "task_context_count": len(task_files),
        "missing_task_contexts": missing,
        "checkpoint": str(checkpoint),
        "checkpoint_exists": checkpoint.is_dir(),
        "output_root": str(_output_root(args)),
        "selected_tasks": _selected_tasks(args),
        "episodes_per_task": args.num_episodes,
        "api_key_present": bool(os.getenv("OPENAI_API_KEY")),
    }
    if MODEL != "gpt-5-nano":
        raise RuntimeError("This evaluation is locked to gpt-5-nano.")
    if missing:
        raise RuntimeError(f"Missing task contexts: {missing}")
    if not checkpoint.is_dir():
        raise RuntimeError(f"GroundSG checkpoint not found: {checkpoint}")
    if args.num_episodes != 2:
        print(f"Warning: requested {args.num_episodes} episodes instead of the configured 2.")
    return checks


class EpisodeEvaluator:
    def __init__(self, args: Args, output_root: Path) -> None:
        self.args = args
        self.output_root = output_root
        self.client = websocket_client_policy.MMEVLAWebsocketClientPolicy(args.host, args.port)
        self.planner = GPT5NanoGroundSGPlanner(output_root / "planner_logs")

    def close(self) -> None:
        if self.client is None:
            return
        try:
            if self.args.shutdown_server:
                self.client.shutdown_server()
        except Exception as error:
            print(f"[ours] Could not shut down policy server: {error}")
        finally:
            self.client.close()
            self.client = None

    def _reset_policy(self) -> None:
        response = self.client.reset()
        while not response.get("reset_finished", False):
            time.sleep(0.1)

    def _action_chunk(
        self,
        state: EpisodeState,
        image: np.ndarray,
        wrist_image: np.ndarray,
        robot_state: np.ndarray,
        task_goal: str,
        grounded_subgoal: str,
    ) -> list[np.ndarray]:
        response = self.client.add_buffer(pack_buffer(state.image_buffer, state.state_buffer, state.exec_start_idx))
        while not response.get("add_buffer_finished", False):
            time.sleep(0.1)

        request = {
            "observation/image": image,
            "observation/wrist_image": wrist_image,
            "observation/state": robot_state,
            "prompt": task_goal,
            "simple_subgoal": grounded_subgoal,
            "grounded_subgoal": grounded_subgoal,
        }
        actions = self.client.infer(request)["actions"]
        if len(actions) < PLANNER_INTERVAL:
            raise RuntimeError(f"pi0.5 returned {len(actions)} actions; at least {PLANNER_INTERVAL} required.")
        return actions[:PLANNER_INTERVAL]

    def run_episode(
        self,
        env_runner: EnvRunner,
        video_directory: Path,
    ) -> tuple[str, dict[str, Any], FailureAudit | None]:
        self._reset_policy()
        pre_traj = env_runner.get_init_obs()
        task_goal = pre_traj["task_goal"]
        recorder = RolloutRecorder(video_directory, task_goal, fps=30)
        state = EpisodeState()
        state.image_buffer.extend(pre_traj["images"])
        state.wrist_image_buffer.extend(pre_traj["wrist_images"])
        state.state_buffer.extend(pre_traj["states"])
        state.exec_start_idx = len(state.image_buffer) - 1

        for index, image in enumerate(pre_traj["images"]):
            recorder.record(
                image=image.copy(),
                wrist_image=pre_traj["wrist_images"][index].copy(),
                state=pre_traj["states"][index].copy(),
                is_video_demo=(env_runner.env_id in TASK_WITH_VIDEO_DEMO and index < len(pre_traj["images"]) - 1),
                subgoal="[initializing...]",
            )

        grounded = self.planner.start_episode(
            task_name=env_runner.env_id,
            episode_id=env_runner.episode_id,
            task_goal=task_goal,
            initial_frames=pre_traj["images"],
        )
        image, wrist_image, robot_state = state.get_current_obs()
        outcome = "unknown"
        diagnostic_trace: list[dict[str, Any]] = []
        _append_changed(
            diagnostic_trace,
            _diagnostic_snapshot(env_runner.info, state.count),
        )
        recent_frames = list(pre_traj["images"][-FRAME_COUNT:])

        while state.count < self.args.max_steps:
            actions = self._action_chunk(
                state,
                image,
                wrist_image,
                robot_state,
                task_goal,
                grounded.text,
            )
            state.clear_buffers()
            planner_frames: list[np.ndarray] = []

            for action in actions:
                observation, stop, outcome = env_runner.step(action)
                state.count += 1
                if observation[0] is None:
                    outcome = "error"
                    stop = True
                    break

                image, wrist_image, robot_state = observation
                state.add_observation(image, wrist_image, robot_state)
                planner_frames.append(image.copy())
                recent_frames.append(image.copy())
                recent_frames = recent_frames[-FRAME_COUNT:]
                recorder.record(
                    image=image.copy(),
                    wrist_image=wrist_image.copy(),
                    state=robot_state.copy(),
                    action=np.asarray(action).copy(),
                    subgoal=grounded.text,
                )
                _append_changed(
                    diagnostic_trace,
                    _diagnostic_snapshot(env_runner.info, state.count),
                )
                if stop or state.count >= self.args.max_steps:
                    break

            if outcome in {"success", "fail", "timeout", "error"}:
                break
            if state.count >= self.args.max_steps:
                outcome = "timeout"
                break
            if len(planner_frames) != PLANNER_INTERVAL:
                raise RuntimeError(f"Expected {PLANNER_INTERVAL} new frames, received {len(planner_frames)}.")
            grounded = self.planner.update(
                _sample_four(planner_frames),
                current_step=state.count,
            )

        if outcome == "unknown":
            outcome = "timeout"

        filename = f"{env_runner.env_id}_ep{env_runner.episode_id}_{outcome}_{task_goal}_{env_runner.difficulty}.mp4"
        recorder.save_video(filename)

        audit = None
        if outcome in {"fail", "timeout"} and self.args.audit_failures:
            audit = self.planner.audit_failure(
                outcome=outcome,
                diagnostic_trace=diagnostic_trace,
                final_frames=recent_frames,
            )

        details = {
            "task": env_runner.env_id,
            "episode": env_runner.episode_id,
            "difficulty": env_runner.difficulty,
            "task_goal": task_goal,
            "outcome": outcome,
            "success": outcome == "success",
            "execution_steps": state.count,
            "demo_frame_count": max(len(pre_traj["images"]) - 1, 0),
            "exec_start_idx": len(pre_traj["images"]) - 1,
            "planner_calls": self.planner.call_index,
            "final_grounded_subgoal": grounded.text,
            "diagnostic_trace": diagnostic_trace,
            "failure_audit": audit.model_dump(mode="json") if audit else None,
        }
        return outcome, details, audit


def _update_failure_csv(
    path: Path,
    task: str,
    episode: int,
    outcome: str,
    audit: FailureAudit | None,
) -> None:
    rows: list[dict[str, Any]] = []
    if path.is_file():
        with path.open(newline="", encoding="utf-8-sig") as stream:
            rows = list(csv.DictReader(stream))
    rows = [row for row in rows if not (row.get("task") == task and row.get("episode(0~9)") == str(episode))]
    if audit is not None:
        rows.append(
            {
                "suite": POLICY_NAME,
                "task": task,
                "episode(0~9)": episode,
                "outcome": outcome,
                "first_failure_step": audit.first_failure_step,
                "primary_failure": audit.primary_failure,
                "secondary_failure": audit.secondary_failure or "",
                "confidence": audit.confidence,
                "notes": audit.notes,
            }
        )

    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=FAILURE_COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def _task_report(task: str, episode_results: dict[str, Any]) -> dict[str, Any]:
    rows = [value for key, value in episode_results.get(task, {}).items() if key.isdigit()]
    counts = {name: sum(row["outcome"] == name for row in rows) for name in ("success", "fail", "timeout", "error")}
    failure_counts: dict[str, int] = {}
    for row in rows:
        audit = row.get("failure_audit")
        if not audit:
            continue
        label = audit["primary_failure"]
        failure_counts[label] = failure_counts.get(label, 0) + 1
    return {
        "task": task,
        "episodes": len(rows),
        **counts,
        "success_rate": counts["success"] / len(rows) if rows else 0.0,
        "primary_failure_counts": failure_counts,
        "episode_results": rows,
    }


def evaluate(args: Args) -> None:
    checks = validate_setup(args)
    print(json.dumps(checks, indent=2))
    if args.dry_run:
        print("[ours] Dry run passed; no server connection or API call was made.")
        return
    if not os.getenv("OPENAI_API_KEY"):
        raise RuntimeError("OPENAI_API_KEY is not set.")

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
        for task in _selected_tasks(args):
            progress.setdefault(task, {})
            episode_results.setdefault(task, {})
            env_runner = EnvRunner(task, video_directory, max_steps=args.max_steps)
            try:
                count = min(args.num_episodes, env_runner.num_episodes)
                for episode in range(count):
                    key = str(episode)
                    previous = episode_results[task].get(key)
                    if previous and previous.get("outcome") != "error":
                        print(f"[ours] {task} ep{episode} already complete; skipping.")
                        continue
                    env_runner.make_env(episode)
                    print(f"\n[ours] Starting {task} ep{episode} with {MODEL}")
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
                        print(f"[ours] ERROR {task} ep{episode}: {error}")
                    finally:
                        env_runner.close_env()

                    progress[task][key] = outcome == "success"
                    episode_results[task][key] = details
                    _save_json(progress_path, progress)
                    _save_json(episode_results_path, episode_results)
                    _update_failure_csv(
                        failure_csv,
                        task,
                        episode,
                        outcome,
                        audit if outcome != "success" else None,
                    )
                    print(
                        f"[ours] Finished {task} ep{episode}: {outcome}; "
                        f"failure={audit.primary_failure if audit else 'none'}"
                    )
            finally:
                env_runner.close_env()

            report = _task_report(task, episode_results)
            with report_jsonl.open("a", encoding="utf-8") as stream:
                stream.write(json.dumps(report, ensure_ascii=False) + "\n")
            _save_json(output_root / "task_summaries" / f"{task}.json", report)
            print(
                f"\n[ours] TASK COMPLETE {task}: "
                f"{report['success']}/{report['episodes']} success "
                f"({report['success_rate']:.1%})"
            )
            time.sleep(1)

        reports = {
            task: _task_report(task, episode_results) for task in _selected_tasks(args) if task in episode_results
        }
        log = {
            "model": MODEL,
            "planner_interval": PLANNER_INTERVAL,
            "success_rate": {task: report["success_rate"] for task, report in reports.items()},
        }
        values = list(log["success_rate"].values())
        log["total_success_rate"] = sum(values) / len(values) if values else 0.0
        _save_json(output_root / "log.json", log)
    finally:
        evaluator.close()


if __name__ == "__main__":
    import tyro

    tyro.cli(evaluate)
