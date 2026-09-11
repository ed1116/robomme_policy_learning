"""Offline autoregressive ButtonUnmaskSwap evaluation on successful oracle videos."""

from __future__ import annotations

from collections import Counter
import dataclasses
import json
from pathlib import Path
import traceback
from typing import Any

from subgoal_prediction.object_memory.oracle_video import FRAME_OFFSETS
from subgoal_prediction.object_memory.oracle_video import LoadedEpisode
from subgoal_prediction.object_memory.oracle_video import load_episode
from subgoal_prediction.object_memory.oracle_video import parse_episode
from subgoal_prediction.object_memory.oracle_video import select_successive_successes
from subgoal_prediction.object_memory.oracle_video import target_colors
from subgoal_prediction.object_memory.planner import SplitObjectMemoryPlanner
from subgoal_prediction.object_memory.planners import DEFAULT_MODEL_ID
from subgoal_prediction.object_memory.planners import MODEL_SLUG
from subgoal_prediction.object_memory.planners import GemmaBackend
from subgoal_prediction.object_memory.schemas import BoundaryComparison

DEFAULT_VIDEO_DIR = Path(
    "runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/oracle/videos"
)
DEFAULT_OUTPUT_DIR = Path(
    f"runs/evaluation/object-memory-offline/{MODEL_SLUG}/ButtonUnmaskSwap_ep3-7"
)


@dataclasses.dataclass
class Args:
    video_dir: str = str(DEFAULT_VIDEO_DIR)
    output_dir: str = str(DEFAULT_OUTPUT_DIR)
    model_id_or_path: str = DEFAULT_MODEL_ID
    episode_count: int = 5
    only_episode: int = -1
    max_calls_per_episode: int = 0
    aggregate_shards_dir: str = ""
    dry_run: bool = False


def _write_json(path: Path, value: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _oracle_label(stage: str, colors: list[str]) -> str:
    labels = {
        "press_first_button": "press the first button",
        "press_second_button": "press the second button",
        "pick_first_target_container": f"pick up the container hiding the {colors[0]} cube",
        "put_down_container": "put down the container",
        "pick_second_target_container": (
            f"pick up the container hiding the {colors[1]} cube"
            if len(colors) > 1
            else "invalid second target"
        ),
        "remain_static": "remain static",
    }
    return labels[stage]


def _canonical_predicted_stage(
    subgoal: dict[str, Any],
    colors: list[str],
) -> str | None:
    raw_stage = subgoal.get("stage")
    if not isinstance(raw_stage, str) or not raw_stage.strip():
        return None
    canonical = {
        "press_first_button",
        "press_second_button",
        "pick_first_target_container",
        "put_down_container",
        "pick_second_target_container",
        "remain_static",
    }
    if raw_stage in canonical:
        return raw_stage

    text = " ".join(raw_stage.lower().replace("_", " ").split())
    if "press" in text and "first button" in text:
        return "press_first_button"
    if "press" in text and "second button" in text:
        return "press_second_button"
    if "put down" in text and "container" in text:
        return "put_down_container"
    if "remain static" in text:
        return "remain_static"
    if "pick" in text and "container" in text:
        target_color = subgoal.get("target_color")
        if not isinstance(target_color, str):
            target_color = next((color for color in colors if color in text), None)
        if colors and target_color == colors[0]:
            return "pick_first_target_container"
        if len(colors) > 1 and target_color == colors[1]:
            return "pick_second_target_container"
    return raw_stage


def _associated_colors(memory: dict[str, Any], container_id: str) -> set[str]:
    states = {item["id"]: item for item in memory["states"]}
    colors: set[str] = set()
    for event in memory["events"]:
        if (
            event["object_id"] == container_id
            and event["type"] in {"covered", "uncovered"}
            and event["subject_id"] in states
        ):
            color = states[event["subject_id"]].get("color")
            if color:
                colors.add(color)
    for relation in memory["relations"]:
        if relation["subject_id"] == container_id and relation["object_id"] in states:
            color = states[relation["object_id"]].get("color")
            if color:
                colors.add(color)
    return colors


def _target_failure(record: dict[str, Any], oracle_stage: str, colors: list[str]) -> str | None:
    subgoal = record["decision"]["next_subgoal"]
    target_id = subgoal["target_entity_id"]
    if oracle_stage in {"put_down_container", "remain_static"}:
        return None
    entities = {item["id"]: item for item in record["memory_after"]["entities"]}
    target = entities.get(target_id)
    if target is None:
        return "unresolved_target_entity"
    expected_type = "button" if oracle_stage.startswith("press_") else "container"
    if target["type"] != expected_type:
        return "wrong_target_type"
    if record["grounded_subgoal"]["point_yx"] is None:
        return "target_has_no_grounding"
    if oracle_stage.startswith("pick_"):
        color_index = 0 if oracle_stage == "pick_first_target_container" else 1
        expected_color = colors[color_index]
        associations = _associated_colors(record["memory_after"], target_id)
        if expected_color not in associations:
            return "target_container_identity_unverified"
    return None


def _compare(
    episode: LoadedEpisode,
    boundary_index: int,
    record: dict[str, Any] | None,
    error: str | None,
) -> BoundaryComparison:
    boundary = episode.boundaries[boundary_index]
    if error is not None or record is None:
        return BoundaryComparison(
            call_frame=boundary.call_frame,
            oracle_stage=boundary.oracle_stage,
            predicted_stage=None,
            oracle_subgoal_completed=boundary.oracle_subgoal_completed,
            predicted_subgoal_completed=None,
            matches=False,
            failure_reason="model_call_error",
        )
    decision = record["decision"]
    colors = target_colors(episode.spec.task_goal)
    predicted_stage = _canonical_predicted_stage(
        decision["next_subgoal"],
        colors,
    )
    predicted_completed = decision["subgoal_completed"]
    reason = None
    if predicted_completed != boundary.oracle_subgoal_completed:
        reason = (
            "premature_subgoal_completion"
            if predicted_completed
            else "late_subgoal_completion"
        )
    elif predicted_stage != boundary.oracle_stage:
        reason = "wrong_subgoal"
    elif record["structural_warning"]:
        reason = "invalid_subgoal_transition"
    else:
        reason = _target_failure(
            record,
            boundary.oracle_stage,
            colors,
        )
    return BoundaryComparison(
        call_frame=boundary.call_frame,
        oracle_stage=boundary.oracle_stage,
        predicted_stage=predicted_stage,
        oracle_subgoal_completed=boundary.oracle_subgoal_completed,
        predicted_subgoal_completed=predicted_completed,
        matches=reason is None,
        failure_reason=reason,
    )


def _episode_manifest(episode: LoadedEpisode) -> dict[str, Any]:
    colors = target_colors(episode.spec.task_goal)
    return {
        "episode_id": episode.spec.episode_id,
        "video": str(episode.spec.path),
        "source_outcome": episode.spec.outcome,
        "difficulty": episode.spec.difficulty,
        "task_goal": episode.spec.task_goal,
        "frame_count": episode.frame_count,
        "fps": episode.fps,
        "oracle_text_transition_frames": episode.oracle_text_transition_frames,
        "oracle_completion_frames": episode.oracle_completion_frames,
        "oracle_stages": episode.oracle_stages,
        "boundaries": [
            {
                **item.model_dump(mode="json"),
                "oracle_subgoal": _oracle_label(item.oracle_stage, colors),
            }
            for item in episode.boundaries
        ],
    }


def aggregate_shards(root: Path) -> dict[str, Any]:
    reports = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("shard_ep*/ep*/episode_report.json"))
    ]
    if not reports:
        raise RuntimeError(f"No completed shard reports found under {root}")
    call_records = [
        json.loads(path.read_text(encoding="utf-8"))
        for path in sorted(root.glob("shard_ep*/ep*/call_t*/call.json"))
    ]
    event_counts = Counter(
        event["type"]
        for call in call_records
        for event in call["current_call_events"]
    )
    stage_counts = Counter(
        call["decision"]["next_subgoal"]["stage"] for call in call_records
    )
    episode_rows = []
    for report in sorted(reports, key=lambda item: item["episode_id"]):
        first_failure = next(
            (item for item in report["calls"] if not item["comparison"]["matches"]),
            None,
        )
        episode_rows.append(
            {
                "episode_id": report["episode_id"],
                "success": report["success"],
                "first_divergence_frame": report["first_divergence_frame"],
                "oracle_change_frame": (
                    first_failure["oracle_change_frame"]
                    if first_failure is not None
                    else None
                ),
                "failure_reason": report["failure_reason"],
                "matching_calls": report["matching_calls"],
                "total_calls": report["total_calls"],
            }
        )
    summary = {
        "model": MODEL_SLUG,
        "architecture": "perception_vlm -> deterministic_reducer -> decision_vlm",
        "call_interval": 16,
        "input_frame_offsets": list(FRAME_OFFSETS),
        "episode_count": len(reports),
        "successful_episodes": sum(item["success"] for item in reports),
        "matching_calls": sum(item["matching_calls"] for item in reports),
        "total_calls": sum(item["total_calls"] for item in reports),
        "model_call_errors": len(list(root.glob("shard_ep*/ep*/call_t*/error.json"))),
        "nonempty_perception_deltas": sum(
            bool(call["perception"]["observed_objects"]) for call in call_records
        ),
        "predicted_subgoal_completions": sum(
            call["decision"]["subgoal_completed"] for call in call_records
        ),
        "event_counts": dict(event_counts),
        "predicted_stage_counts": dict(stage_counts),
        "structural_warnings": sum(
            bool(call["structural_warning"]) for call in call_records
        ),
        "episodes": episode_rows,
    }
    _write_json(root / "summary.json", summary)
    return summary


def evaluate(args: Args) -> None:
    if args.aggregate_shards_dir:
        summary = aggregate_shards(Path(args.aggregate_shards_dir))
        print(json.dumps(summary, indent=2))
        return
    video_dir = Path(args.video_dir)
    output_dir = Path(args.output_dir)
    if not video_dir.is_dir():
        raise RuntimeError(f"Oracle video directory not found: {video_dir}")
    if args.only_episode >= 0:
        matching = [
            parse_episode(path)
            for path in video_dir.glob(f"ButtonUnmaskSwap_ep{args.only_episode}_*.mp4")
        ]
        if len(matching) != 1 or matching[0].outcome != "success":
            raise RuntimeError(
                f"Episode {args.only_episode} is missing, ambiguous, or not successful"
            )
        specs = matching
        selection_rule = f"explicit successful oracle episode {args.only_episode}"
    else:
        specs = select_successive_successes(video_dir, args.episode_count)
        selection_rule = (
            f"first {args.episode_count} successive successful oracle episodes"
        )
    episodes = [load_episode(spec) for spec in specs]
    manifest = {
        "selection_rule": selection_rule,
        "episode_ids": [item.spec.episode_id for item in episodes],
        "episodes": [_episode_manifest(item) for item in episodes],
    }
    _write_json(output_dir / "oracle_manifest.json", manifest)
    print(f"[object-memory] Selected episodes: {manifest['episode_ids']}")
    for episode in episodes:
        print(
            f"[object-memory] ep{episode.spec.episode_id}: "
            f"{episode.frame_count} frames, "
            f"text_transitions={episode.oracle_text_transition_frames}, "
            f"completions={episode.oracle_completion_frames}"
        )
    if args.dry_run:
        print("[object-memory] Dry run complete; no model was loaded.")
        return

    backend = GemmaBackend(args.model_id_or_path)
    planner = SplitObjectMemoryPlanner(backend)
    episode_reports: list[dict[str, Any]] = []
    for episode in episodes:
        planner.reset()
        episode_dir = output_dir / f"ep{episode.spec.episode_id}"
        calls: list[dict[str, Any]] = []
        print(f"[object-memory] Replaying ep{episode.spec.episode_id}")
        boundaries = (
            episode.boundaries[: args.max_calls_per_episode]
            if args.max_calls_per_episode > 0
            else episode.boundaries
        )
        for boundary_index, boundary in enumerate(boundaries):
            frame_indices, frames = episode.window(boundary.call_frame)
            call_dir = episode_dir / f"call_t{boundary.call_frame:04d}"
            record = None
            error = None
            try:
                record = planner.run_cycle(
                    task_goal=episode.spec.task_goal,
                    call_frame=boundary.call_frame,
                    frame_indices=frame_indices,
                    frames=frames,
                    call_dir=call_dir,
                )
            except Exception as caught:
                error = f"{type(caught).__name__}: {caught}"
                _write_json(
                    call_dir / "error.json",
                    {"error": error, "traceback": traceback.format_exc()},
                )
            comparison = _compare(episode, boundary_index, record, error)
            call_summary = {
                "call_frame": boundary.call_frame,
                "oracle_change_frame": boundary.oracle_change_frame,
                "oracle_subgoal": _oracle_label(
                    boundary.oracle_stage,
                    target_colors(episode.spec.task_goal),
                ),
                "comparison": comparison.model_dump(mode="json"),
                "predicted_grounded_subgoal": (
                    record["grounded_subgoal"] if record is not None else None
                ),
                "error": error,
                "log_dir": str(call_dir),
            }
            calls.append(call_summary)
            status = "match" if comparison.matches else comparison.failure_reason
            print(
                f"  ep{episode.spec.episode_id} t={boundary.call_frame:04d}: "
                f"oracle={boundary.oracle_stage}, status={status}"
            )

        first_failure = next((item for item in calls if not item["comparison"]["matches"]), None)
        report = {
            "episode_id": episode.spec.episode_id,
            "success": first_failure is None,
            "first_divergence_frame": (
                first_failure["call_frame"] if first_failure is not None else None
            ),
            "failure_reason": (
                first_failure["comparison"]["failure_reason"]
                if first_failure is not None
                else None
            ),
            "matching_calls": sum(item["comparison"]["matches"] for item in calls),
            "total_calls": len(calls),
            "calls": calls,
        }
        _write_json(episode_dir / "episode_report.json", report)
        episode_reports.append(report)

    summary = {
        "model": MODEL_SLUG,
        "architecture": "perception_vlm -> deterministic_reducer -> decision_vlm",
        "call_interval": 16,
        "input_frame_offsets": list(FRAME_OFFSETS),
        "episode_count": len(episode_reports),
        "successful_episodes": sum(item["success"] for item in episode_reports),
        "episodes": [
            {
                key: report[key]
                for key in (
                    "episode_id",
                    "success",
                    "first_divergence_frame",
                    "failure_reason",
                    "matching_calls",
                    "total_calls",
                )
            }
            for report in episode_reports
        ],
    }
    _write_json(output_dir / "summary.json", summary)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    import tyro

    tyro.cli(evaluate)
