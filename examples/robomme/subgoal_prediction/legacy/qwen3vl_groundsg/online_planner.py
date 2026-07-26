from __future__ import annotations

from collections.abc import Iterable
import copy
import json
import os
from pathlib import Path
import re
import tempfile
from typing import Any

os.environ.setdefault("VIDEO_MAX_TOKEN_NUM", "64")
os.environ.setdefault("FPS_MAX_FRAMES", "10")

import cv2
import numpy as np
from pydantic import BaseModel
from pydantic import ValidationError
from subgoal_prediction.legacy.gpt5nano_groundsg.schemas import FailureAudit
from subgoal_prediction.legacy.gpt5nano_groundsg.schemas import GroundedSubgoal
from subgoal_prediction.legacy.gpt5nano_groundsg.schemas import PlannerOutput
from subgoal_prediction.legacy.gpt5nano_groundsg.schemas import Subgoal
from subgoal_prediction.legacy.gpt5nano_groundsg.schemas import UpdatedMemory
from swift.llm import InferRequest
from swift.llm import PtEngine
from swift.llm import RequestConfig
import torch

MODEL = "qwen3-vl-4b-instruct"
MODEL_REPO = "Qwen/Qwen3-VL-4B-Instruct"
DEFAULT_MODEL_DIR = Path("runs/ckpts/vlm_subgoal_predictor/qwenvl/Qwen3-VL-4B-Instruct")
FRAME_COUNT = 4
FRAME_STRIDE = 4
CONTEXT_SPAN = FRAME_COUNT * FRAME_STRIDE
RECENT_EVENT_LIMIT = 32
TASK_VISUAL_HINTS = {
    "BinFill": (
        "Look specifically for small solid colored cubes, the small gray square button, "
        "and the larger gray open receptacle with a dark interior. The receptacle is "
        "opaque_bin_a, not an aperture box."
    ),
    "InsertPeg": ("The peg insertion destination with a visible opening is aperture_box_a, not an opaque bin."),
}


class QwenPlannerOutput(PlannerOutput):
    grounded_subgoal: GroundedSubgoal


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


def _extract_json(raw_output: str) -> dict[str, Any]:
    text = raw_output.strip()
    fenced = re.fullmatch(r"```(?:json)?\s*(.*?)\s*```", text, flags=re.DOTALL | re.IGNORECASE)
    if fenced:
        text = fenced.group(1).strip()
    decoder = json.JSONDecoder()
    for match in re.finditer(r"\{", text):
        try:
            value, end = decoder.raw_decode(text[match.start() :])
        except json.JSONDecodeError:
            continue
        if text[match.start() + end :].strip():
            raise ValueError("Unexpected text after the JSON object.")
        if not isinstance(value, dict):
            raise ValueError("Expected a JSON object.")
        return value
    raise ValueError("No JSON object was found in the model output.")


def parse_model_output(raw_output: str, schema: type[BaseModel]) -> BaseModel:
    return schema.model_validate(_extract_json(raw_output))


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


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


def _response_text(response: Any) -> str:
    choice = response[0].choices[0] if isinstance(response, list) else response.choices[0]
    content = choice.message.content
    if not isinstance(content, str):
        raise RuntimeError(f"Unexpected Swift response content: {type(content).__name__}")
    return content


class Qwen3VLGroundSGPlanner:
    """Stateful base-instruct Qwen planner backed by one reusable Swift PtEngine."""

    def __init__(
        self,
        output_root: Path,
        *,
        model_dir: Path | str = DEFAULT_MODEL_DIR,
        max_retries: int = 3,
        max_tokens: int = 4096,
        engine: Any | None = None,
    ) -> None:
        self.output_root = Path(output_root)
        self.output_root.mkdir(parents=True, exist_ok=True)
        self.model_dir = Path(model_dir).resolve()
        if not self.model_dir.is_dir():
            raise RuntimeError(f"Qwen checkpoint not found: {self.model_dir}")
        self.max_retries = max_retries
        self.max_tokens = max_tokens
        self.engine = engine if engine is not None else self._load_engine()

        gpt_package = Path(__file__).resolve().parents[1] / "gpt5nano_groundsg"
        self.base_rules = (gpt_package / "planner_rules.txt").read_text(encoding="utf-8")
        self.task_context_root = gpt_package / "task_contexts"
        self.reset()

    def _load_engine(self) -> Any:
        print(f"[qwen] Loading base instruct model once from {self.model_dir}")
        return PtEngine(
            model_id_or_path=str(self.model_dir),
            adapters=None,
            attn_impl="sdpa",
            torch_dtype=torch.float16,
        )

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

    def _initialize_episode(self, task_name: str, episode_id: int, task_goal: str) -> None:
        self.reset()
        self.task_name = task_name
        self.episode_id = episode_id
        self.task_goal = task_goal
        self.task_context = self._load_task_context(task_name)
        self.episode_dir = self.output_root / task_name / f"ep{episode_id}"
        self.episode_dir.mkdir(parents=True, exist_ok=True)

    def start_episode(
        self,
        *,
        task_name: str,
        episode_id: int,
        task_goal: str,
        initial_frames: list[np.ndarray],
    ) -> GroundedSubgoal:
        self._initialize_episode(task_name, episode_id, task_goal)

        frames = [np.asarray(frame, dtype=np.uint8) for frame in initial_frames]
        if not frames:
            raise RuntimeError("Environment reset returned no front-camera frames.")
        self.exec_start_idx = len(frames) - 1

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

        return self._call(
            _sample_four(frames),
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

    def run_saved_call(
        self,
        *,
        task_name: str,
        task_goal: str,
        frames: list[np.ndarray],
        episode_id: int = 0,
    ) -> GroundedSubgoal:
        """Run one execution-phase call against an already sampled four-frame window."""
        if len(frames) != FRAME_COUNT:
            raise RuntimeError(f"Expected exactly four saved frames, got {len(frames)}.")
        self._initialize_episode(task_name, episode_id, task_goal)
        return self._call(
            frames,
            current_step=0,
            execution_step=0,
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

    def _system_prompt(self) -> str:
        visual_hint = TASK_VISUAL_HINTS.get(self.task_name, "Use only direct visual evidence for object inventory.")
        return (
            f"{self.base_rules}\n\n"
            "QWEN SINGLE-PASS OUTPUT\n"
            "- Perform visual object detection, memory update, completion judgment, "
            "and grounded-subgoal prediction together in this one response.\n"
            "- Before selecting a subgoal, inventory every visible allowed task object "
            "across the four images. "
            "An empty updated_memory is invalid when any task object is visible.\n"
            "- Put tight 256x256 [y_min,x_min,y_max,x_max] boundaries directly in "
            "updated_memory.states. Bound each visible object itself, not a loose region around it.\n"
            "- In addition to the shared PlannerOutput fields, return grounded_subgoal. "
            "It must be new_subgoal with every <> replaced by the corresponding target "
            "bbox center [y,x], in order, where y=(y_min+y_max)//2 and "
            "x=(x_min+x_max)//2. Its points_yx and target_object_ids must match.\n"
            "- Return one JSON object containing subgoal_completed, updated_memory, "
            "new_subgoal, and grounded_subgoal. grounded_subgoal contains text, "
            "target_object_ids, grounding_required, and points_yx.\n"
            f"- Task-specific visual prior: {visual_hint}"
        )

    @staticmethod
    def _user_prompt(request: dict[str, Any]) -> str:
        images = "\n".join(
            f"Chronological frame {index}/4 ({offset}): <image>"
            for index, offset in enumerate(("t-12", "t-8", "t-4", "t"), start=1)
        )
        return (
            "Use the four images below in chronological order.\n"
            "For a grounded subgoal, new_subgoal.text must contain the literal slot <> "
            "and target_object_ids must contain the matching canonical entity ID. Return "
            "the coordinate-resolved grounded_subgoal in the same JSON response.\n"
            f"{images}\n\nINPUT PAYLOAD\n"
            f"{json.dumps(request, indent=2, ensure_ascii=False)}"
        )

    @staticmethod
    def _messages(
        system_prompt: str,
        user_prompt: str,
        prior_raw: str | None = None,
        validation_error: str | None = None,
    ) -> list[dict[str, str]]:
        correction = ""
        if prior_raw is not None and validation_error is not None:
            correction = (
                "\n\nCORRECTION REQUIRED\n"
                "The previous output below was invalid. Do not repeat it. Reinspect all four "
                "images and fix the cause, not only the syntax. Every <> slot needs one "
                "target_object_id, and that ID needs a matching entity and state with a "
                "tight bbox estimated from the supplied images. "
                "In new_subgoal.text, write the two literal characters <> "
                "for each coordinate slot; never put an ID or coordinate between them. The "
                "grounded_subgoal must replace those slots with the exact bbox centers. "
                "Return the complete corrected JSON object only.\n"
                f"PREVIOUS INVALID OUTPUT\n{prior_raw}\n"
                f"VALIDATION ERROR\n{validation_error}"
            )
        return [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt + correction},
        ]

    def _infer(
        self,
        *,
        messages: list[dict[str, str]],
        image_paths: list[str],
        max_tokens: int | None = None,
    ) -> str:
        response = self.engine.infer(
            [InferRequest(messages=messages, images=image_paths)],
            request_config=RequestConfig(
                max_tokens=max_tokens or self.max_tokens,
                temperature=0,
                seed=7,
            ),
        )
        return _response_text(response)

    def _validate_transition(
        self,
        candidate: QwenPlannerOutput,
        *,
        phase: str,
        execution_step: int | None,
        current_step: int,
    ) -> GroundedSubgoal:
        if phase == "demonstration" and candidate.subgoal_completed:
            raise ValueError("Planner marked a demonstration-phase subgoal complete.")
        if phase == "execution" and execution_step == 0 and candidate.subgoal_completed:
            raise ValueError("Planner marked the first execution subgoal complete.")

        previous_entities = {item["id"]: item["type"] for item in self.memory.get("entities", [])}
        candidate_entities = {item.id: item.type for item in candidate.updated_memory.entities}
        missing = sorted(set(previous_entities) - set(candidate_entities))
        renamed = sorted(
            entity_id
            for entity_id, entity_type in previous_entities.items()
            if candidate_entities.get(entity_id) not in {None, entity_type}
        )
        if missing:
            raise ValueError(f"Updated memory dropped persistent entity IDs: {missing}")
        if renamed:
            raise ValueError(f"Updated memory changed persistent entity types: {renamed}")

        visible_states = [state for state in candidate.updated_memory.states if state.visibility == "visible"]
        if not visible_states:
            raise ValueError("Updated memory contains no visible task objects.")
        unlocalized = [state.id for state in visible_states if state.bbox_yxyx is None]
        if unlocalized:
            raise ValueError(f"Visible states require tight bboxes: {unlocalized}")
        stale = [state.id for state in visible_states if state.last_seen_step != current_step]
        if stale:
            raise ValueError(f"Directly visible states must refresh last_seen_step to {current_step}: {stale}")

        if (
            phase == "execution"
            and execution_step is not None
            and execution_step > 0
            and not candidate.subgoal_completed
            and self.active
            and candidate.new_subgoal.model_dump(mode="json") != self.active
        ):
            raise ValueError("Incomplete subgoal changed identity or text.")
        expected = ground_subgoal(candidate.updated_memory, candidate.new_subgoal)
        if (
            candidate.grounded_subgoal.target_object_ids != expected.target_object_ids
            or candidate.grounded_subgoal.grounding_required != expected.grounding_required
        ):
            raise ValueError("grounded_subgoal targets do not match new_subgoal.")
        candidate.grounded_subgoal = expected
        return expected

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
        call_prefix = f"call_t{step_label:04d}"
        frame_offsets = ("t_minus_12", "t_minus_8", "t_minus_4", "t")
        image_paths: list[str] = []
        for index, (frame, offset) in enumerate(zip(frames, frame_offsets, strict=True), start=1):
            filename = f"{call_prefix}_image_{index}_{offset}.png"
            path = self.episode_dir / filename
            if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                raise RuntimeError(f"Could not save planner input frame: {path}")
            image_paths.append(str(path.resolve()))

        system_prompt = self._system_prompt()
        user_prompt = self._user_prompt(request)
        input_path = self.episode_dir / f"{call_prefix}_input.json"
        output_path = self.episode_dir / f"{call_prefix}_output.json"

        result: QwenPlannerOutput | None = None
        grounded_result: GroundedSubgoal | None = None
        previous_raw: str | None = None
        previous_error: str | None = None
        for _ in range(self.max_retries):
            messages = self._messages(
                system_prompt,
                user_prompt,
                previous_raw,
                previous_error,
            )
            _write_json(input_path, {"messages": messages, "images": image_paths})
            raw_output = self._infer(
                messages=messages,
                image_paths=image_paths,
            )
            try:
                logged_output: Any = _extract_json(raw_output)
            except ValueError:
                logged_output = raw_output
            _write_json(output_path, logged_output)
            try:
                candidate = parse_model_output(raw_output, QwenPlannerOutput)
                assert isinstance(candidate, QwenPlannerOutput)
                candidate_grounded = self._validate_transition(
                    candidate,
                    phase=phase,
                    execution_step=execution_step,
                    current_step=current_step,
                )
                result = candidate
                grounded_result = candidate_grounded
            except (AssertionError, RuntimeError, ValidationError, ValueError) as error:
                previous_raw = raw_output
                previous_error = str(error)
            if result is not None:
                break

        if result is None or grounded_result is None:
            raise RuntimeError(f"Qwen planner failed after {self.max_retries} attempts: {previous_error}")

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

        trace_record = {
            "call_index": self.call_index,
            "model": MODEL,
            "phase": phase,
            "current_step": current_step,
            "execution_step": execution_step,
            "subgoal_completed": result.subgoal_completed,
            "active_subgoal": copy.deepcopy(self.active),
            "grounded_subgoal": grounded_result.model_dump(mode="json"),
            "completed_subgoals": copy.deepcopy(self.completed),
            "updated_memory": self.memory,
        }
        self.trace.append(trace_record)
        self.call_index += 1
        return grounded_result

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
                "Use uncertain when evidence cannot separate perception, planning, and control."
            ),
        }
        system_prompt = (
            "You audit RoboMME failures using only the supplied trace and final frames. "
            "Return one JSON object with first_failure_step, primary_failure, "
            "secondary_failure, confidence, and notes. No prose."
        )
        user_prompt = (
            "\n".join(f"Final frame {index}/4: <image>" for index in range(1, 5))
            + "\n\n"
            + json.dumps(prompt, indent=2, ensure_ascii=False)
        )
        error = "not attempted"
        with tempfile.TemporaryDirectory(prefix="qwen_failure_audit_") as directory:
            image_paths: list[str] = []
            for index, frame in enumerate(_pad_four(final_frames), start=1):
                path = Path(directory) / f"frame_{index}.png"
                if not cv2.imwrite(str(path), cv2.cvtColor(frame, cv2.COLOR_RGB2BGR)):
                    raise RuntimeError(f"Could not save temporary failure-audit frame: {path}")
                image_paths.append(str(path))
            try:
                messages = self._messages(system_prompt, user_prompt)
                raw = self._infer(
                    messages=messages,
                    image_paths=image_paths,
                    max_tokens=512,
                )
                audit = parse_model_output(raw, FailureAudit)
                assert isinstance(audit, FailureAudit)
                return audit
            except (AssertionError, RuntimeError, ValidationError, ValueError) as caught:
                error = str(caught)
        return FailureAudit(
            first_failure_step=None,
            primary_failure="uncertain",
            secondary_failure=None,
            confidence="low",
            notes=f"Automatic Qwen failure audit was unavailable or invalid: {error}",
        )


__all__ = [
    "CONTEXT_SPAN",
    "DEFAULT_MODEL_DIR",
    "FRAME_COUNT",
    "FRAME_STRIDE",
    "MODEL",
    "MODEL_REPO",
    "Qwen3VLGroundSGPlanner",
    "QwenPlannerOutput",
    "_four_frame_windows",
    "_pad_four",
    "_sample_four",
    "ground_subgoal",
    "parse_model_output",
]
