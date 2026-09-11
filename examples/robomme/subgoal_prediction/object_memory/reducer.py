from __future__ import annotations

import math
import string

from .schemas import EntityState
from .schemas import MemoryEntity
from .schemas import MemoryEvent
from .schemas import MemoryRelation
from .schemas import MemoryState
from .schemas import ObjectObservation
from .schemas import PerceptionOutput

MOVE_DEADBAND_NORM1000 = 20
NEAREST_MATCH_PIXELS = 45.0
RECENT_EVENT_LIMIT = 96


def _center(bbox: list[int]) -> tuple[float, float]:
    y1, x1, y2, x2 = bbox
    return ((y1 + y2) / 2, (x1 + x2) / 2)


def _distance(left: list[int], right: list[int]) -> float:
    ly, lx = _center(left)
    ry, rx = _center(right)
    return math.hypot(ly - ry, lx - rx)


def _bbox_moved(previous: list[int], current: list[int]) -> bool:
    return any(
        abs(previous_value - current_value) >= MOVE_DEADBAND_NORM1000
        for previous_value, current_value in zip(previous, current, strict=True)
    )


def _center_is_inside(inner: list[int], outer: list[int]) -> bool:
    center_y, center_x = _center(inner)
    y1, x1, y2, x2 = outer
    return y1 <= center_y <= y2 and x1 <= center_x <= x2


def _to_image_bbox(normalized_xyxy: list[int]) -> list[int]:
    x1, y1, x2, y2 = normalized_xyxy
    return [
        min(256, max(0, round(value * 256 / 1000)))
        for value in (y1, x1, y2, x2)
    ]


def _to_normalized_bbox(image_yxyx: list[int]) -> list[int]:
    y1, x1, y2, x2 = image_yxyx
    return [
        round(value * 1000 / 256)
        for value in (x1, y1, x2, y2)
    ]


def _suffix(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    quotient, remainder = divmod(index, len(string.ascii_lowercase))
    return f"{string.ascii_lowercase[quotient - 1]}{string.ascii_lowercase[remainder]}"


class MemoryReducer:
    """Own deterministic identity, relations, events, and persistent state."""

    def __init__(self) -> None:
        self.state = MemoryState()

    def reset(self) -> None:
        self.state = MemoryState()

    def _next_id(self, entity_type: str, reserved: set[str]) -> str:
        index = 0
        while f"{entity_type}_{_suffix(index)}" in reserved:
            index += 1
        return f"{entity_type}_{_suffix(index)}"

    @staticmethod
    def _best_automatic_match(
        observation: ObjectObservation,
        entities: dict[str, MemoryEntity],
        states: dict[str, EntityState],
        claimed: set[str],
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for entity_id, entity in entities.items():
            if entity_id in claimed or entity.type != observation.type:
                continue
            distance = _distance(
                _to_image_bbox(observation.bbox_xyxy_norm1000),
                states[entity_id].bbox_yxyx,
            )
            if distance <= NEAREST_MATCH_PIXELS:
                candidates.append((distance, entity_id))
        return min(candidates)[1] if candidates else None

    def _resolve_observations(
        self,
        observations: list[ObjectObservation],
        entities: dict[str, MemoryEntity],
        states: dict[str, EntityState],
    ) -> list[tuple[ObjectObservation, str]]:
        claimed: set[str] = set()
        resolved: list[tuple[ObjectObservation, str]] = []
        pending: list[ObjectObservation] = []

        for observation in observations:
            requested = observation.entity_id
            if (
                requested in entities
                and entities[requested].type == observation.type
                and requested not in claimed
            ):
                entity_id = requested
            else:
                entity_id = self._best_automatic_match(
                    observation,
                    entities,
                    states,
                    claimed,
                )
            if entity_id is None:
                pending.append(observation)
                continue
            claimed.add(entity_id)
            resolved.append((observation, entity_id))

        reserved = set(entities) | claimed
        pending.sort(
            key=lambda item: (
                item.type,
                _center(_to_image_bbox(item.bbox_xyxy_norm1000))[1],
            )
        )
        for observation in pending:
            entity_id = self._next_id(observation.type, reserved)
            reserved.add(entity_id)
            claimed.add(entity_id)
            resolved.append((observation, entity_id))
        return resolved

    @staticmethod
    def _event_key(event: MemoryEvent) -> tuple[int, str, str, str | None]:
        return event.frame, event.type, event.subject_id, event.object_id

    def _append_event(self, events: list[MemoryEvent], event: MemoryEvent) -> None:
        if self._event_key(event) not in {self._event_key(item) for item in events}:
            events.append(event)

    def _remove_cover_and_record_uncovered(
        self,
        *,
        cube_id: str,
        evidence_frame: int,
        relations: dict[tuple[str, str, str], MemoryRelation],
        events: list[MemoryEvent],
        uncertainties: list[str],
    ) -> None:
        covering_keys = [
            key
            for key, relation in relations.items()
            if relation.relation == "covers" and relation.object_id == cube_id
        ]
        for key in covering_keys:
            relations.pop(key)
        if len(covering_keys) == 1:
            container_id = covering_keys[0][0]
            self._append_event(
                events,
                MemoryEvent(
                    frame=evidence_frame,
                    type="uncovered",
                    subject_id=cube_id,
                    object_id=container_id,
                    source="reducer",
                    detail="The previously occluded cube became directly visible.",
                ),
            )
        elif len(covering_keys) > 1:
            uncertainties.append(
                f"{cube_id} became visible but had multiple covering relations; "
                "all were removed without selecting one uncovered container."
            )

    def _apply_strict_occlusions(
        self,
        *,
        perception: PerceptionOutput,
        observed_entity_ids: set[str],
        container_observations: dict[str, tuple[list[int], int]],
        entities: dict[str, MemoryEntity],
        states: dict[str, EntityState],
        relations: dict[tuple[str, str, str], MemoryRelation],
        events: list[MemoryEvent],
        uncertainties: list[str],
    ) -> None:
        candidate_map: dict[str, list[str]] = {}
        evidence_frames: dict[str, int] = {}

        for change in perception.observed_visibility_changes:
            entity_id = change.entity_id
            evidence_frames[entity_id] = change.evidence_frame
            if entity_id not in entities:
                uncertainties.append(
                    f"Ignored visibility change for unknown entity {entity_id}."
                )
                continue
            if entity_id in observed_entity_ids:
                uncertainties.append(
                    f"Ignored contradictory visible and occluded reports for {entity_id}."
                )
                continue
            state = states[entity_id]
            if state.visibility != "visible":
                uncertainties.append(
                    f"Ignored repeated occlusion for {entity_id}; "
                    f"previous visibility was {state.visibility}."
                )
                continue
            if entities[entity_id].type != "cube":
                state.visibility = "occluded"
                state.motion = "unknown"
                continue

            candidates = [
                container_id
                for container_id, (container_bbox, _) in container_observations.items()
                if not states[container_id].held
                and _center_is_inside(state.bbox_yxyx, container_bbox)
                and not any(
                    relation.subject_id == container_id
                    and relation.relation == "covers"
                    and relation.object_id != entity_id
                    for relation in relations.values()
                )
            ]
            candidate_map[entity_id] = candidates

        reverse_candidates: dict[str, list[str]] = {}
        for cube_id, container_ids in candidate_map.items():
            for container_id in container_ids:
                reverse_candidates.setdefault(container_id, []).append(cube_id)

        for cube_id, candidates in candidate_map.items():
            state = states[cube_id]
            unambiguous = (
                len(candidates) == 1
                and len(reverse_candidates[candidates[0]]) == 1
            )
            if not unambiguous:
                state.visibility = "unobserved"
                state.motion = "unknown"
                uncertainties.append(
                    f"Could not validate a unique covering container for {cube_id}; "
                    f"candidates={candidates}."
                )
                continue

            container_id = candidates[0]
            evidence_frame = max(
                evidence_frames[cube_id],
                container_observations[container_id][1],
            )
            relation_key = (container_id, "covers", cube_id)
            relations[relation_key] = MemoryRelation(
                subject_id=container_id,
                relation="covers",
                object_id=cube_id,
                last_evidence_frame=evidence_frame,
            )
            state.visibility = "occluded"
            state.motion = "unknown"
            self._append_event(
                events,
                MemoryEvent(
                    frame=evidence_frame,
                    type="covered",
                    subject_id=cube_id,
                    object_id=container_id,
                    source="reducer",
                    detail=(
                        "The cube became occluded and its previous bbox center "
                        "is inside one uniquely matched container bbox."
                    ),
                ),
            )

    def apply(self, perception: PerceptionOutput, current_frame: int) -> MemoryState:
        observation_ids = [
            observation.observation_id
            for observation in perception.observed_objects
        ]
        if len(observation_ids) != len(set(observation_ids)):
            raise ValueError("observation_id must be unique within one response")

        entities = {
            item.id: item.model_copy(deep=True)
            for item in self.state.entities
        }
        states = {
            item.id: item.model_copy(deep=True)
            for item in self.state.states
        }
        relations = {
            (item.subject_id, item.relation, item.object_id): item.model_copy(deep=True)
            for item in self.state.relations
        }
        events = [item.model_copy(deep=True) for item in self.state.events]
        uncertainties = list(perception.uncertainties)

        for state in states.values():
            state.motion = "stationary" if state.visibility == "visible" else "unknown"

        resolved = self._resolve_observations(
            perception.observed_objects,
            entities,
            states,
        )
        observed_entity_ids = {entity_id for _, entity_id in resolved}
        container_observations: dict[str, tuple[list[int], int]] = {}

        for observation, entity_id in resolved:
            image_bbox = _to_image_bbox(observation.bbox_xyxy_norm1000)
            previous = states.get(entity_id)
            if previous is None:
                entities[entity_id] = MemoryEntity(
                    id=entity_id,
                    type=observation.type,
                )
                states[entity_id] = EntityState(
                    id=entity_id,
                    bbox_yxyx=image_bbox,
                    present=True,
                    visibility="visible",
                    motion="stationary",
                    held=observation.held is True,
                    pressed=observation.pressed,
                    highlighted=observation.highlighted,
                    color=observation.color,
                    last_seen_frame=observation.evidence_frame,
                )
                self._append_event(
                    events,
                    MemoryEvent(
                        frame=observation.evidence_frame,
                        type="appeared",
                        subject_id=entity_id,
                        object_id=None,
                        source="reducer",
                        detail="First direct observation.",
                    ),
                )
            else:
                previous_visibility = previous.visibility
                previous_pressed = previous.pressed
                previous_held = previous.held
                previous_normalized_bbox = _to_normalized_bbox(previous.bbox_yxyx)
                if _bbox_moved(
                    previous_normalized_bbox,
                    observation.bbox_xyxy_norm1000,
                ):
                    previous.motion = "moving"
                    previous.bbox_yxyx = image_bbox
                    self._append_event(
                        events,
                        MemoryEvent(
                            frame=observation.evidence_frame,
                            type="moved",
                            subject_id=entity_id,
                            object_id=None,
                            source="reducer",
                            detail=(
                                "At least one normalized xyxy coordinate changed "
                                "by 20 units or more."
                            ),
                        ),
                    )
                else:
                    previous.motion = "stationary"

                previous.color = observation.color or previous.color
                previous.present = True
                previous.visibility = "visible"
                if observation.pressed is not None:
                    previous.pressed = observation.pressed
                if observation.held is not None:
                    previous.held = observation.held
                if observation.highlighted is not None:
                    previous.highlighted = observation.highlighted
                previous.last_seen_frame = observation.evidence_frame

                if (
                    observation.type == "button"
                    and observation.pressed is True
                    and previous_pressed is not True
                ):
                    self._append_event(
                        events,
                        MemoryEvent(
                            frame=observation.evidence_frame,
                            type="pressed",
                            subject_id=entity_id,
                            object_id=None,
                            source="reducer",
                            detail="The observed button state changed to pressed.",
                        ),
                    )
                if observation.held is True and not previous_held:
                    self._append_event(
                        events,
                        MemoryEvent(
                            frame=observation.evidence_frame,
                            type="picked",
                            subject_id=entity_id,
                            object_id=None,
                            source="reducer",
                            detail="The observed held state changed to true.",
                        ),
                    )
                elif observation.held is False and previous_held:
                    self._append_event(
                        events,
                        MemoryEvent(
                            frame=observation.evidence_frame,
                            type="placed",
                            subject_id=entity_id,
                            object_id=None,
                            source="reducer",
                            detail="The observed held state changed to false.",
                        ),
                    )
                if observation.type == "cube" and previous_visibility == "occluded":
                    self._remove_cover_and_record_uncovered(
                        cube_id=entity_id,
                        evidence_frame=observation.evidence_frame,
                        relations=relations,
                        events=events,
                        uncertainties=uncertainties,
                    )

            if observation.type == "container":
                container_observations[entity_id] = (
                    image_bbox,
                    observation.evidence_frame,
                )

        self._apply_strict_occlusions(
            perception=perception,
            observed_entity_ids=observed_entity_ids,
            container_observations=container_observations,
            entities=entities,
            states=states,
            relations=relations,
            events=events,
            uncertainties=uncertainties,
        )

        holding = next((item.id for item in states.values() if item.held), None)
        self.state = MemoryState(
            current_frame=current_frame,
            entities=sorted(entities.values(), key=lambda item: item.id),
            states=sorted(states.values(), key=lambda item: item.id),
            relations=sorted(
                relations.values(),
                key=lambda item: (item.subject_id, item.object_id),
            ),
            events=events[-RECENT_EVENT_LIMIT:],
            robot={"holding": holding},
            uncertainties=uncertainties,
        )
        return self.state.model_copy(deep=True)
