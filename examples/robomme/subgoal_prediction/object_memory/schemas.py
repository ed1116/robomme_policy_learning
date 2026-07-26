from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import model_validator

EntityType = Literal["button", "container", "cube"]
EventType = Literal["appeared", "moved", "covered", "uncovered", "pressed", "picked", "placed"]
Stage = Literal[
    "press_first_button",
    "press_second_button",
    "pick_first_target_container",
    "put_down_container",
    "pick_second_target_container",
    "remain_static",
]


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectObservation(StrictModel):
    observation_id: str
    entity_id: str | None = None
    type: EntityType
    bbox_yxyx: list[int] = Field(min_length=4, max_length=4)
    color: str | None = None
    pressed: bool | None = None
    held: bool = False
    evidence_frame: int
    confidence: float = Field(ge=0.0, le=1.0)

    @model_validator(mode="after")
    def validate_bbox(self) -> ObjectObservation:
        y1, x1, y2, x2 = self.bbox_yxyx
        if not (0 <= y1 < y2 <= 256 and 0 <= x1 < x2 <= 256):
            raise ValueError("bbox_yxyx must be ordered inside the 256x256 front view")
        return self


class RelationObservation(StrictModel):
    subject_ref: str
    relation: Literal["covers"]
    object_ref: str
    evidence_frame: int
    confidence: float = Field(ge=0.0, le=1.0)


class EventObservation(StrictModel):
    type: EventType
    subject_ref: str
    object_ref: str | None = None
    evidence_frame: int
    confidence: float = Field(ge=0.0, le=1.0)
    detail: str


class PerceptionOutput(StrictModel):
    observed_objects: list[ObjectObservation]
    observed_relations: list[RelationObservation]
    observed_events: list[EventObservation]
    uncertainties: list[str]

    @model_validator(mode="after")
    def validate_observation_ids(self) -> PerceptionOutput:
        identifiers = [item.observation_id for item in self.observed_objects]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("observation_id must be unique within one response")
        return self


class EntityState(StrictModel):
    id: str
    type: EntityType
    bbox_yxyx: list[int]
    color: str | None
    visibility: Literal["visible", "unobserved"]
    pressed: bool | None
    held: bool
    first_seen_frame: int
    last_seen_frame: int
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryRelation(StrictModel):
    subject_id: str
    relation: Literal["covers"]
    object_id: str
    last_evidence_frame: int
    confidence: float = Field(ge=0.0, le=1.0)


class MemoryEvent(StrictModel):
    frame: int
    type: EventType
    subject_id: str
    object_id: str | None
    confidence: float = Field(ge=0.0, le=1.0)
    source: Literal["vlm", "reducer"]
    detail: str


class RobotMemory(StrictModel):
    holding: str | None = None


class MemoryState(StrictModel):
    current_frame: int = 0
    entities: list[EntityState] = Field(default_factory=list)
    relations: list[MemoryRelation] = Field(default_factory=list)
    events: list[MemoryEvent] = Field(default_factory=list)
    robot: RobotMemory = Field(default_factory=RobotMemory)
    uncertainties: list[str] = Field(default_factory=list)


class SubgoalDecision(StrictModel):
    stage: Stage
    target_entity_id: str | None
    target_color: str | None

    @model_validator(mode="after")
    def validate_target(self) -> SubgoalDecision:
        needs_target = self.stage not in {"put_down_container", "remain_static"}
        if needs_target and self.target_entity_id is None:
            raise ValueError(f"{self.stage} requires target_entity_id")
        if not needs_target and self.target_entity_id is not None:
            raise ValueError(f"{self.stage} cannot have target_entity_id")
        pick_stage = self.stage in {"pick_first_target_container", "pick_second_target_container"}
        if not pick_stage and self.target_color is not None:
            raise ValueError("target_color is only valid for a target-container pickup")
        return self


class DecisionOutput(StrictModel):
    subgoal_completed: bool
    completion_evidence: str
    next_subgoal: SubgoalDecision
    confidence: float = Field(ge=0.0, le=1.0)


class GroundedSubgoal(StrictModel):
    stage: Stage
    text: str
    target_entity_id: str | None
    point_yx: list[int] | None


class OracleBoundary(StrictModel):
    call_frame: int
    oracle_stage: Stage
    oracle_subgoal_completed: bool
    oracle_change_frame: int | None


class BoundaryComparison(StrictModel):
    call_frame: int
    oracle_stage: Stage
    predicted_stage: Stage | None
    oracle_subgoal_completed: bool
    predicted_subgoal_completed: bool | None
    matches: bool
    failure_reason: str | None
