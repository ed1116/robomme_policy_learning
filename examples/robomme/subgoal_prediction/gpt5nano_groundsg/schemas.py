import re
from typing import Literal

from pydantic import BaseModel
from pydantic import Field
from pydantic import model_validator

EntityType = Literal[
    "cube",
    "peg",
    "hook stick",
    "routing stick",
    "target",
    "button",
    "container",
    "aperture box",
    "opaque bin",
]


class Entity(BaseModel):
    id: str
    type: EntityType

    @model_validator(mode="after")
    def validate_canonical_id(self) -> "Entity":
        prefix = self.type.replace(" ", "_")
        if re.fullmatch(rf"{re.escape(prefix)}_[a-z]+", self.id) is None:
            raise ValueError(f"{self.type} IDs must use {prefix}_a, {prefix}_b, ...")
        return self


class ObjectState(BaseModel):
    id: str
    bbox_yxyx: list[int] | None = Field(default=None, min_length=4, max_length=4)
    present: bool
    visibility: Literal["visible", "occluded", "out_of_view", "unknown"]
    motion: Literal["stationary", "moving", "unknown"]
    held: bool
    pressed: bool | None = None
    highlighted: bool | None = None
    color: str | None = None
    last_seen_step: int | None = None

    @model_validator(mode="after")
    def validate_bbox(self) -> "ObjectState":
        if self.bbox_yxyx is not None:
            y1, x1, y2, x2 = self.bbox_yxyx
            if not (0 <= y1 < y2 <= 256 and 0 <= x1 < x2 <= 256):
                raise ValueError("bbox_yxyx must be ordered within the 256x256 image.")
        return self


class Relation(BaseModel):
    subject: str
    relation: Literal["covers", "inside", "on"]
    object: str


class RobotState(BaseModel):
    gripper: Literal["open", "closed", "unknown"]
    holding: str | None


class Event(BaseModel):
    step: int
    type: Literal[
        "appeared",
        "disappeared",
        "moved",
        "covered",
        "uncovered",
        "pressed",
        "picked",
        "placed",
        "highlighted",
        "unhighlighted",
    ]
    subject: str
    object: str | None
    detail: str


class UpdatedMemory(BaseModel):
    entities: list[Entity]
    states: list[ObjectState]
    relations: list[Relation]
    robot: RobotState
    events: list[Event]
    uncertainties: list[str]

    @model_validator(mode="after")
    def validate_references(self) -> "UpdatedMemory":
        entity_ids = [entity.id for entity in self.entities]
        state_ids = [state.id for state in self.states]
        if len(entity_ids) != len(set(entity_ids)):
            raise ValueError("Entity IDs must be unique.")
        if len(state_ids) != len(set(state_ids)):
            raise ValueError("State IDs must be unique.")
        if set(state_ids) != set(entity_ids):
            raise ValueError("Every entity requires exactly one state with the same ID.")

        known = set(entity_ids)
        for relation in self.relations:
            if relation.subject not in known or relation.object not in known:
                raise ValueError("Relation references must be existing entity IDs.")
        for event in self.events:
            if event.subject not in known or (event.object is not None and event.object not in known):
                raise ValueError("Event references must be existing entity IDs.")
        if self.robot.holding is not None and self.robot.holding not in known:
            raise ValueError("Robot holding must reference an existing entity ID.")
        return self


class Subgoal(BaseModel):
    text: str
    target_object_ids: list[str]
    grounding_required: bool

    @model_validator(mode="after")
    def validate_grounding_slots(self) -> "Subgoal":
        slot_count = self.text.count("<>")
        if self.grounding_required:
            if not self.target_object_ids:
                raise ValueError("A grounded subgoal requires target_object_ids.")
            if slot_count != len(self.target_object_ids):
                raise ValueError("Each target_object_id requires one ordered <> slot.")
        elif self.target_object_ids or slot_count:
            raise ValueError("An ungrounded subgoal cannot contain targets or <> slots.")
        return self


class GroundedSubgoal(BaseModel):
    text: str
    target_object_ids: list[str]
    grounding_required: bool
    points_yx: list[list[int]]


class PlannerOutput(BaseModel):
    subgoal_completed: Literal[0, 1]
    updated_memory: UpdatedMemory
    new_subgoal: Subgoal


FailureType = Literal[
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


class FailureAudit(BaseModel):
    first_failure_step: int | None
    primary_failure: FailureType
    secondary_failure: FailureType | None
    confidence: Literal["low", "medium", "high"]
    notes: str
