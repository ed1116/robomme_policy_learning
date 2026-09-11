from __future__ import annotations

from typing import Literal

from pydantic import BaseModel
from pydantic import ConfigDict
from pydantic import Field
from pydantic import field_validator
from pydantic import model_validator

EntityType = Literal["button", "container", "cube"]
EventType = Literal["appeared", "moved", "covered", "uncovered", "pressed", "picked", "placed"]
Stage = str


class StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class ObjectObservation(StrictModel):
    observation_id: str
    entity_id: str | None = None
    type: EntityType
    bbox_xyxy_norm1000: list[int] = Field(min_length=4, max_length=4)
    color: str | None = None
    pressed: bool | None = None
    highlighted: bool | None = None
    held: bool | None = None
    evidence_frame: int

    @field_validator("observation_id", mode="before")
    @classmethod
    def normalize_observation_id(cls, value: object) -> object:
        if isinstance(value, int) and not isinstance(value, bool):
            return f"observation_{value}"
        if isinstance(value, str) and value.isdigit():
            return f"observation_{value}"
        return value

    @model_validator(mode="after")
    def validate_bbox(self) -> ObjectObservation:
        x1, y1, x2, y2 = self.bbox_xyxy_norm1000
        if not (0 <= x1 < x2 <= 1000 and 0 <= y1 < y2 <= 1000):
            raise ValueError("bbox_xyxy_norm1000 must be an ordered normalized box")
        return self


class VisibilityObservation(StrictModel):
    entity_id: str
    visibility: Literal["occluded"]
    evidence_frame: int


class PerceptionOutput(StrictModel):
    observed_objects: list[ObjectObservation]
    observed_visibility_changes: list[VisibilityObservation]
    uncertainties: list[str]

    @model_validator(mode="after")
    def validate_observation_ids(self) -> PerceptionOutput:
        identifiers = [item.observation_id for item in self.observed_objects]
        if len(identifiers) != len(set(identifiers)):
            raise ValueError("observation_id must be unique within one response")
        visibility_ids = [
            item.entity_id
            for item in self.observed_visibility_changes
        ]
        if len(visibility_ids) != len(set(visibility_ids)):
            raise ValueError(
                "visibility entity_id must be unique within one response"
            )
        return self


class InitialPerceptionOutput(PerceptionOutput):
    @model_validator(mode="after")
    def validate_initial_inventory(self) -> InitialPerceptionOutput:
        if not self.observed_objects:
            raise ValueError("The initial call requires a non-empty visible-object inventory")
        if self.observed_visibility_changes:
            raise ValueError("The initial call cannot change visibility of unknown entities")
        return self


class MemoryEntity(StrictModel):
    id: str
    type: EntityType


class EntityState(StrictModel):
    id: str
    bbox_yxyx: list[int]
    present: bool
    visibility: Literal["visible", "occluded", "unobserved"]
    motion: Literal["stationary", "moving", "unknown"]
    held: bool
    pressed: bool | None
    highlighted: bool | None
    color: str | None
    last_seen_frame: int


class MemoryRelation(StrictModel):
    subject_id: str
    relation: Literal["covers"]
    object_id: str
    last_evidence_frame: int


class MemoryEvent(StrictModel):
    frame: int
    type: EventType
    subject_id: str
    object_id: str | None
    source: Literal["vlm", "reducer"]
    detail: str


class RobotMemory(StrictModel):
    holding: str | None = None


class MemoryState(StrictModel):
    current_frame: int = 0
    entities: list[MemoryEntity] = Field(default_factory=list)
    states: list[EntityState] = Field(default_factory=list)
    relations: list[MemoryRelation] = Field(default_factory=list)
    events: list[MemoryEvent] = Field(default_factory=list)
    robot: RobotMemory = Field(default_factory=RobotMemory)
    uncertainties: list[str] = Field(default_factory=list)


class SubgoalDecision(StrictModel):
    stage: Stage
    target_entity_id: str | None
    target_color: str | None


class DecisionOutput(StrictModel):
    subgoal_completed: bool
    completion_evidence: str
    next_subgoal: SubgoalDecision


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
