from __future__ import annotations

import base64
from collections.abc import Iterable
import copy
import json
from pathlib import Path
import time
from typing import Any

import cv2
import numpy as np
from openai import OpenAI

from .schemas import FailureAudit
from .schemas import GroundedSubgoal
from .schemas import PlannerOutput
from .schemas import Subgoal
from .schemas import UpdatedMemory

MODEL = "gpt-5-nano"
FRAME_COUNT = 4
FRAME_STRIDE = 4
CONTEXT_SPAN = FRAME_COUNT * FRAME_STRIDE
RECENT_EVENT_LIMIT = 32


def _jsonable(value: Any) -> Any:
    if hasattr(value, "model_dump"):
        return _jsonable(value.model_dump(mode="json"))
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


def _image_data_url(image: np.ndarray) -> str:
    image = np.asarray(image, dtype=np.uint8)
    ok, encoded = cv2.imencode(".png", cv2.cvtColor(image, cv2.COLOR_RGB2BGR))
    if not ok:
        raise RuntimeError("Could not encode planner frame.")
    payload = base64.b64encode(encoded.tobytes()).decode("ascii")
    return f"data:image/png;base64,{payload}"


def _pad_four(frames: Iterable[np.ndarray]) -> list[np.ndarray]:
    result = [np.asarray(frame, dtype=np.uint8) for frame in frames]
    if not result:
        raise RuntimeError("At least one planner frame is required.")
    result = result[-FRAME_COUNT:]
    while len(result) < FRAME_COUNT:
        result.insert(0, result[0].copy())
    return result


def _sample_four(frames: list[np.ndarray], end: int | None = None) -> list[np.ndarray]:
    if not frames:
        raise RuntimeError("At least one planner frame is required.")
    if end is None:
        end = len(frames) - 1
    indices = [max(0, end - offset) for offset in (12, 8, 4, 0)]
    return [np.asarray(frames[index], dtype=np.uint8) for index in indices]


def _four_frame_windows(frames: list[np.ndarray]) -> list[list[np.ndarray]]:
    if not frames:
        return []
    end_indices = [0, *range(CONTEXT_SPAN, len(frames), CONTEXT_SPAN)]
    return [_sample_four(frames, end) for end in end_indices]


def ground_subgoal(memory: UpdatedMemory, subgoal: Subgoal) -> GroundedSubgoal:
    if not subgoal.grounding_required:
        return GroundedSubgoal(
            text=subgoal.text,
            target_object_ids=[],
            grounding_required=False,
            points_yx=[],
        )
    text = subgoal.text.strip()
    points: list[list[int]] = []
    for target_object_id in subgoal.target_object_ids:
        state = next((item for item in memory.states if item.id == target_object_id), None)
        if state is None or state.bbox_yxyx is None:
            raise RuntimeError(f"No usable bbox for target {target_object_id!r}.")
        y1, x1, y2, x2 = state.bbox_yxyx
        point = [(y1 + y2) // 2, (x1 + x2) // 2]
        points.append(point)
        text = text.replace("<>", f"<{point[0]}, {point[1]}>", 1)
    return GroundedSubgoal(
        text=text,
        target_object_ids=subgoal.target_object_ids,
        grounding_required=True,
        points_yx=points,
    )


class GPT5NanoGroundSGPlanner:
    """Stateful GPT-5 Nano planner. The model name is intentionally not configurable."""

    def __init__(
        self,
        output_root: Path,
        *,
        request_timeout: float = 120.0,
        max_retries: int = 4,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.client = OpenAI(timeout=request_timeout, max_retries=0)
        self.max_retries = max_retries
        package_root = Path(__file__).resolve().parent
        self.base_rules = (package_root / "planner_rules.txt").read_text(encoding="utf-8")
        self.task_context_root = package_root / "task_contexts"
        self.reset()

    def reset(self) -> None:
        self.task_name = ""
        self.episode_id = -1
        self.task_goal = ""
        self.task_context = ""
        self.episode_dir: Path | None = None
        self.memory: dict[str, Any] = {}
        self.active: dict[str, Any] = {}
        self.completed: list[dict[str, Any]] = []
        self.trace: list[dict[str, Any]] = []
        self.grounded_subgoal: GroundedSubgoal | None = None
        self.call_index = 0
        self.exec_start_idx = 0

    def _load_task_context(self, task_name: str) -> str:
        path = self.task_context_root / f"{task_name}.txt"
        if not path.is_file():
            raise RuntimeError(f"Missing task context: {path}")
        return path.read_text(encoding="utf-8")

    def start_episode(
        self,
        *,
        task_name: str,
        episode_id: int,
        task_goal: str,
        initial_frames: list[np.ndarray],
    ) -> GroundedSubgoal:
        self.reset()
        self.task_name = task_name
        self.episode_id = episode_id
        self.task_goal = task_goal
        self.task_context = self._load_task_context(task_name)
        self.episode_dir = self.output_root / task_name / f"ep{episode_id}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

        frames = [np.asarray(frame, dtype=np.uint8) for frame in initial_frames]
        if not frames:
            raise RuntimeError("Environment reset returned no front-camera frames.")

        self.exec_start_idx = len(frames) - 1

        # Every reset-provided frame before the last one is observation-only.
        demo_frames = frames[:-1]
        if demo_frames:
            for observation_step, window in zip(
                [0, *range(CONTEXT_SPAN, len(demo_frames), CONTEXT_SPAN)],
                _four_frame_windows(demo_frames),
                strict=True,
            ):
                self._call(
                    window,
                    current_step=observation_step,
                    execution_step=None,
                    phase="demonstration",
                )

        execution_window = _sample_four(frames)
        return self._call(
            execution_window,
            current_step=self.exec_start_idx,
            execution_step=0,
            phase="execution",
        )

    def update(self, frames: list[np.ndarray], current_step: int) -> GroundedSubgoal:
        if len(frames) != FRAME_COUNT:
            raise RuntimeError(f"Expected exactly four execution frames, got {len(frames)}.")
        return self._call(
            frames,
            current_step=self.exec_start_idx + current_step,
            execution_step=current_step,
            phase="execution",
        )

    def _request_payload(
        self,
        current_step: int,
        execution_step: int | None,
        phase: str,
    ) -> dict[str, Any]:
        memory = copy.deepcopy(self.memory)
        if memory.get("events"):
            memory["events"] = memory["events"][-RECENT_EVENT_LIMIT:]
        return {
            "main_instruction": self.task_goal,
            "task_name": self.task_name,
            "task_context": self.task_context,
            "phase": phase,
            "current_step": current_step,
            "execution_step": execution_step,
            "demo_frame_count": self.exec_start_idx,
            "frames": [
                {"offset": offset, "description": f"chronological frame {index + 1}/4"}
                for index, offset in enumerate([-12, -8, -4, 0])
            ],
            "completed_subgoals": copy.deepcopy(self.completed),
            "active_subgoal": copy.deepcopy(self.active),
            "past_object_centric_memory": memory,
        }

    def _call(
        self,
        frames: list[np.ndarray],
        *,
        current_step: int,
        execution_step: int | None,
        phase: str,
    ) -> GroundedSubgoal:
        frames = _pad_four(frames)
        request = self._request_payload(current_step, execution_step, phase)
        assert self.episode_dir is not None
        step_label = execution_step if execution_step is not None else current_step
        call_prefix = f"planner_call_{self.call_index:06d}_t{step_label:06d}"
        frame_offsets = ("t_minus_12", "t_minus_8", "t_minus_4", "t")
        frame_files: list[str] = []
        for index, (frame, offset) in enumerate(zip(frames, frame_offsets, strict=True), start=1):
            filename = f"{call_prefix}_image_{index}_{offset}.png"
            path = self.episode_dir / filename
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not save planner input frame: {path}")
            frame_files.append(filename)

        request_record = {
            "call_index": self.call_index,
            "model": MODEL,
            "instructions": self.base_rules,
            "text_format": "PlannerOutput",
            "payload": request,
            "input_images": frame_files,
        }
        (self.episode_dir / f"{call_prefix}_request.json").write_text(
            json.dumps(request_record, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        content: list[dict[str, Any]] = [
            {
                "type": "input_text",
                "text": (
                    "Use the four attached frames in chronological order.\n"
                    + json.dumps(request, indent=2, ensure_ascii=False)
                ),
            }
        ]
        content.extend(
            {"type": "input_image", "image_url": _image_data_url(frame), "detail": "high"} for frame in frames
        )

        result: PlannerOutput | None = None
        grounded_result: GroundedSubgoal | None = None
        response = None
        last_error: Exception | None = None
        attempt_errors: list[dict[str, Any]] = []
        for attempt in range(self.max_retries):
            candidate: PlannerOutput | None = None
            try:
                response = self.client.responses.parse(
                    model=MODEL,
                    instructions=self.base_rules,
                    input=[{"role": "user", "content": content}],
                    text_format=PlannerOutput,
                )
                candidate = response.output_parsed
                if candidate is None:
                    raise RuntimeError("GPT-5 Nano returned no parsed planner output.")
                if phase == "demonstration" and candidate.subgoal_completed:
                    raise RuntimeError("Planner marked a demonstration-phase subgoal complete.")
                if phase == "execution" and execution_step == 0 and candidate.subgoal_completed:
                    raise RuntimeError("Planner marked the first execution subgoal complete.")
                if (
                    phase == "execution"
                    and execution_step is not None
                    and execution_step > 0
                    and not candidate.subgoal_completed
                    and self.active
                    and candidate.new_subgoal.model_dump(mode="json") != self.active
                ):
                    raise RuntimeError("Incomplete subgoal changed identity or text.")

                candidate_grounded = ground_subgoal(candidate.updated_memory, candidate.new_subgoal)
                result = candidate
                grounded_result = candidate_grounded
                break
            except Exception as error:
                last_error = error
                error_record: dict[str, Any] = {"attempt": attempt + 1, "error": repr(error)}
                if candidate is not None:
                    error_record["parsed_output"] = candidate.model_dump(mode="json")
                attempt_errors.append(error_record)
                if attempt + 1 == self.max_retries:
                    break
                time.sleep(min(2**attempt, 8))
        if attempt_errors:
            (self.episode_dir / f"{call_prefix}_attempt_errors.json").write_text(
                json.dumps(attempt_errors, indent=2, ensure_ascii=False),
                encoding="utf-8",
            )
        if result is None or grounded_result is None:
            raise RuntimeError(f"GPT-5 Nano planner failed after retries: {last_error}")

        previous_active = copy.deepcopy(self.active)
        if (
            phase == "execution"
            and execution_step is not None
            and execution_step > 0
            and result.subgoal_completed
            and previous_active
        ):
            self.completed.append(previous_active)
        self.memory = result.updated_memory.model_dump(mode="json")
        self.active = result.new_subgoal.model_dump(mode="json")
        self.grounded_subgoal = grounded_result

        record = {
            "call_index": self.call_index,
            "model": MODEL,
            "api_response_id": getattr(response, "id", None),
            "api_usage": _jsonable(getattr(response, "usage", None)),
            "phase": phase,
            "current_step": current_step,
            "execution_step": execution_step,
            "subgoal_completed": result.subgoal_completed,
            "active_subgoal": copy.deepcopy(self.active),
            "grounded_subgoal": self.grounded_subgoal.model_dump(mode="json"),
            "completed_subgoals": copy.deepcopy(self.completed),
            "updated_memory": self.memory,
        }
        self.trace.append(record)
        path = self.episode_dir / f"{call_prefix}_output.json"
        path.write_text(json.dumps(record, indent=2, ensure_ascii=False), encoding="utf-8")
        (self.episode_dir / "planner_trace.json").write_text(
            json.dumps(self.trace, indent=2, ensure_ascii=False),
            encoding="utf-8",
        )
        self.call_index += 1
        return self.grounded_subgoal

    def audit_failure(
        self,
        *,
        outcome: str,
        diagnostic_trace: list[dict[str, Any]],
        final_frames: list[np.ndarray],
    ) -> FailureAudit:
        taxonomy = [
            "early_advance",
            "late_advance",
            "wrong_localization",
            "wrong_referent",
            "counting_or_memory_error",
            "low_level_execution",
            "environment_error",
            "wrong_semantics",
            "uncertain",
        ]
        compact_trace = [
            {
                "step": item["execution_step"],
                "done": item["subgoal_completed"],
                "subgoal": item["grounded_subgoal"]["text"],
            }
            for item in self.trace
            if item["phase"] == "execution"
        ]
        prompt = {
            "task": self.task_name,
            "episode": self.episode_id,
            "goal": self.task_goal,
            "outcome": outcome,
            "taxonomy": taxonomy,
            "planner_trace": compact_trace,
            "oracle_diagnostic_trace": diagnostic_trace,
            "instructions": (
                "Classify the earliest causal failure. Oracle diagnostics are for post-hoc audit only. "
                "Use uncertain when the evidence cannot separate perception, planning, and control."
            ),
        }
        content: list[dict[str, Any]] = [
            {"type": "input_text", "text": json.dumps(prompt, indent=2, ensure_ascii=False)}
        ]
        content.extend(
            {"type": "input_image", "image_url": _image_data_url(frame), "detail": "high"}
            for frame in _pad_four(final_frames)
        )
        try:
            response = self.client.responses.parse(
                model=MODEL,
                instructions=(
                    "You audit RoboMME failures using only the supplied trace and final frames. "
                    "Return the taxonomy label most directly supported by evidence."
                ),
                input=[{"role": "user", "content": content}],
                text_format=FailureAudit,
            )
            if response.output_parsed is not None:
                return response.output_parsed
        except Exception:
            pass
        return FailureAudit(
            first_failure_step=None,
            primary_failure="uncertain",
            secondary_failure=None,
            confidence="low",
            notes="Automatic GPT-5 Nano failure audit was unavailable or inconclusive.",
        )


__all__ = [
    "CONTEXT_SPAN",
    "FRAME_COUNT",
    "FRAME_STRIDE",
    "MODEL",
    "GPT5NanoGroundSGPlanner",
    "_four_frame_windows",
    "_pad_four",
    "_sample_four",
    "ground_subgoal",
]
