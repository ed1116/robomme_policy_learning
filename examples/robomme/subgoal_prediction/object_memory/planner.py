from __future__ import annotations

import copy
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from .reducer import MemoryReducer
from .reducer import _to_normalized_bbox
from .schemas import DecisionOutput
from .schemas import GroundedSubgoal
from .schemas import InitialPerceptionOutput
from .schemas import MemoryEvent
from .schemas import MemoryState
from .schemas import PerceptionOutput
from .schemas import SubgoalDecision


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _event_key(event: MemoryEvent) -> tuple[int, str, str, str | None]:
    return event.frame, event.type, event.subject_id, event.object_id


class SplitObjectMemoryPlanner:
    """Perception VLM -> deterministic reducer -> decision VLM."""

    def __init__(self, backend: Any, package_root: Path | None = None) -> None:
        self.backend = backend
        root = package_root or Path(__file__).resolve().parent
        shared = (root / "planner_rules.txt").read_text(encoding="utf-8")
        self.initial_perception_prompt = (
            root / "perception_rules_initial.txt"
        ).read_text(encoding="utf-8")
        self.delta_perception_prompt = (
            root / "perception_rules_delta.txt"
        ).read_text(encoding="utf-8")
        self.decision_prompt = shared + "\n\n" + (root / "decision_rules.txt").read_text(encoding="utf-8")
        self.reducer = MemoryReducer()
        self.active_subgoal: SubgoalDecision | None = None
        self.completed_subgoals: list[SubgoalDecision] = []

    def reset(self) -> None:
        self.reducer.reset()
        self.active_subgoal = None
        self.completed_subgoals = []

    @staticmethod
    def _save_images(call_dir: Path, frame_indices: list[int], frames: list[np.ndarray]) -> list[str]:
        paths: list[str] = []
        for index, (frame_index, frame) in enumerate(zip(frame_indices, frames, strict=True), start=1):
            path = call_dir / f"front_{index}_frame_{frame_index:04d}.png"
            bgr = cv2.cvtColor(np.asarray(frame, dtype=np.uint8), cv2.COLOR_RGB2BGR)
            if not cv2.imwrite(str(path), bgr):
                raise RuntimeError(f"Could not save VLM input image: {path}")
            paths.append(str(path.resolve()))
        return paths

    @staticmethod
    def _compact_memory(memory: MemoryState) -> dict[str, Any]:
        value = memory.model_dump(mode="json")
        value["events"] = value["events"][-32:]
        return value

    @staticmethod
    def _track_table(memory: MemoryState) -> list[dict[str, Any]]:
        states = {item.id: item for item in memory.states}
        return [
            {
                "id": item.id,
                "type": item.type,
                "color": states[item.id].color,
                "visibility": states[item.id].visibility,
                "pressed": states[item.id].pressed,
                "held": states[item.id].held,
                "highlighted": states[item.id].highlighted,
                "last_bbox_xyxy_norm1000": _to_normalized_bbox(
                    states[item.id].bbox_yxyx
                ),
                "last_seen_frame": states[item.id].last_seen_frame,
            }
            for item in memory.entities
        ]

    @staticmethod
    def _normalize_evidence_frames(perception: PerceptionOutput, frame_indices: list[int]) -> list[str]:
        corrections: list[str] = []
        allowed = set(frame_indices)
        current = frame_indices[-1]
        for item in [
            *perception.observed_objects,
            *perception.observed_visibility_changes,
        ]:
            if item.evidence_frame not in allowed:
                corrections.append(f"{type(item).__name__}:{item.evidence_frame}->{current}")
                item.evidence_frame = current
        return corrections

    @staticmethod
    def _ground(memory: MemoryState, subgoal: SubgoalDecision) -> GroundedSubgoal:
        stage_text = subgoal.stage.strip().replace("_", " ")
        if subgoal.target_entity_id is None:
            return GroundedSubgoal(
                stage=subgoal.stage,
                text=stage_text.rstrip(".").capitalize() + ".",
                target_entity_id=None,
                point_yx=None,
            )
        state = next((item for item in memory.states if item.id == subgoal.target_entity_id), None)
        if state is None:
            return GroundedSubgoal(
                stage=subgoal.stage,
                text=f"{subgoal.stage} targeting unresolved {subgoal.target_entity_id}.",
                target_entity_id=subgoal.target_entity_id,
                point_yx=None,
            )
        y1, x1, y2, x2 = state.bbox_yxyx
        point = [(y1 + y2) // 2, (x1 + x2) // 2]
        text = f"{stage_text.rstrip('.').capitalize()} at <{point[0]} {point[1]}>."
        return GroundedSubgoal(
            stage=subgoal.stage,
            text=text,
            target_entity_id=subgoal.target_entity_id,
            point_yx=point,
        )

    def run_cycle(
        self,
        *,
        task_goal: str,
        call_frame: int,
        frame_indices: list[int],
        frames: list[np.ndarray],
        call_dir: Path,
    ) -> dict[str, Any]:
        call_dir.mkdir(parents=True, exist_ok=True)
        image_paths = self._save_images(call_dir, frame_indices, frames)
        before_memory = self.reducer.state.model_copy(deep=True)
        is_initial = call_frame == 0
        existing_event_keys = {_event_key(item) for item in before_memory.events}
        perception_payload = {
            "absolute_frame_numbers": frame_indices,
            "object_type_constraint": (
                "There are no objects other than these: ['cube', 'button', 'container']"
            ),
            "existing_tracks_for_identity_matching": self._track_table(before_memory),
            "perception_mode": (
                "initial_visible_object_inventory"
                if is_initial
                else "delta_only_changes_not_inventory"
            ),
        }
        perception_user = (
            "\n".join(
                f"Frame {frame_index}: <image>"
                for frame_index in frame_indices
            )
            + "\n\nINPUT\n"
            + json.dumps(perception_payload, indent=2, ensure_ascii=False)
        )
        perception_schema = InitialPerceptionOutput if is_initial else PerceptionOutput
        perception_prompt = (
            self.initial_perception_prompt
            if is_initial
            else self.delta_perception_prompt
        )
        perception_result = self.backend.infer(
            system_prompt=perception_prompt,
            user_prompt=perception_user,
            image_paths=image_paths,
            schema=perception_schema,
            max_tokens=1600,
        )
        perception = perception_result.parsed
        assert isinstance(perception, PerceptionOutput)
        _write_json(call_dir / "perception_output.json", perception.model_dump(mode="json"))
        _write_json(
            call_dir / "perception_attempts.json",
            {"attempts": perception_result.attempts},
        )
        evidence_corrections = self._normalize_evidence_frames(perception, frame_indices)
        memory = self.reducer.apply(perception, call_frame)
        _write_json(call_dir / "reducer_output.json", memory.model_dump(mode="json"))
        current_events = [
            event.model_dump(mode="json")
            for event in memory.events
            if _event_key(event) not in existing_event_keys
        ]

        decision_payload = {
            "task_goal": task_goal,
            "current_frame": call_frame,
            "button_selection_rule": (
                "If there are two or more buttons left to press, press the one "
                "with the biggest x-coordinate first."
            ),
            "active_subgoal": (
                self.active_subgoal.model_dump(mode="json") if self.active_subgoal is not None else None
            ),
            "completed_subgoals": [
                item.model_dump(mode="json") for item in self.completed_subgoals
            ],
            "current_call_events": current_events,
            "deterministic_object_memory": self._compact_memory(memory),
        }
        decision_result = self.backend.infer(
            system_prompt=self.decision_prompt,
            user_prompt="INPUT\n" + json.dumps(decision_payload, indent=2, ensure_ascii=False),
            image_paths=[],
            schema=DecisionOutput,
            max_tokens=400,
        )
        decision = decision_result.parsed
        assert isinstance(decision, DecisionOutput)
        _write_json(call_dir / "decision_output.json", decision.model_dump(mode="json"))
        _write_json(
            call_dir / "decision_attempts.json",
            {"attempts": decision_result.attempts},
        )

        structural_warning = None
        previous_active = self.active_subgoal.model_copy(deep=True) if self.active_subgoal else None
        if previous_active is None and decision.subgoal_completed:
            structural_warning = "First call cannot complete a previous subgoal."
        elif (
            previous_active is not None
            and not decision.subgoal_completed
            and decision.next_subgoal != previous_active
        ):
            structural_warning = "Model changed the active subgoal without marking it complete."
        if decision.subgoal_completed and previous_active is not None:
            self.completed_subgoals.append(previous_active)
        self.active_subgoal = decision.next_subgoal.model_copy(deep=True)
        grounded = self._ground(memory, self.active_subgoal)
        _write_json(
            call_dir / "grounded_subgoal_output.json",
            grounded.model_dump(mode="json"),
        )

        record = {
            "call_frame": call_frame,
            "frame_indices": frame_indices,
            "perception": perception.model_dump(mode="json"),
            "evidence_frame_corrections": evidence_corrections,
            "memory_before": before_memory.model_dump(mode="json"),
            "memory_after": memory.model_dump(mode="json"),
            "current_call_events": current_events,
            "decision": decision.model_dump(mode="json"),
            "previous_active_subgoal": (
                previous_active.model_dump(mode="json") if previous_active is not None else None
            ),
            "grounded_subgoal": grounded.model_dump(mode="json"),
            "completed_subgoals": [
                item.model_dump(mode="json") for item in self.completed_subgoals
            ],
            "structural_warning": structural_warning,
        }
        _write_json(call_dir / "call.json", record)
        return copy.deepcopy(record)
