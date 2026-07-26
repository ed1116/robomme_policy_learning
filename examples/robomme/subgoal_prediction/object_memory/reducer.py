from __future__ import annotations

import copy
import math
import string

from .schemas import EntityState
from .schemas import EventObservation
from .schemas import MemoryEvent
from .schemas import MemoryRelation
from .schemas import MemoryState
from .schemas import ObjectObservation
from .schemas import PerceptionOutput

MOVE_THRESHOLD_PIXELS = 10.0
NEAREST_MATCH_PIXELS = 45.0
RECENT_EVENT_LIMIT = 96


def _center(bbox: list[int]) -> tuple[float, float]:
    y1, x1, y2, x2 = bbox
    return ((y1 + y2) / 2, (x1 + x2) / 2)


def _distance(left: list[int], right: list[int]) -> float:
    ly, lx = _center(left)
    ry, rx = _center(right)
    return math.hypot(ly - ry, lx - rx)


def _suffix(index: int) -> str:
    if index < len(string.ascii_lowercase):
        return string.ascii_lowercase[index]
    quotient, remainder = divmod(index, len(string.ascii_lowercase))
    return f"{string.ascii_lowercase[quotient - 1]}{string.ascii_lowercase[remainder]}"


class MemoryReducer:
    """Own the complete belief state; the VLM supplies only current observations."""

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
        entities: dict[str, EntityState],
        claimed: set[str],
    ) -> str | None:
        candidates: list[tuple[float, str]] = []
        for entity_id, entity in entities.items():
            if entity_id in claimed or entity.type != observation.type:
                continue
            if observation.color and entity.color and observation.color != entity.color:
                continue
            distance = _distance(observation.bbox_yxyx, entity.bbox_yxyx)
            if distance <= NEAREST_MATCH_PIXELS:
                candidates.append((distance, entity_id))
        return min(candidates)[1] if candidates else None

    def _resolve_observations(
        self,
        observations: list[ObjectObservation],
        entities: dict[str, EntityState],
    ) -> tuple[list[tuple[ObjectObservation, str]], dict[str, str]]:
        claimed: set[str] = set()
        ref_map: dict[str, str] = {}
        resolved: list[tuple[ObjectObservation, str]] = []
        pending: list[ObjectObservation] = []

        for observation in observations:
            requested = observation.entity_id
            if requested in entities and entities[requested].type == observation.type and requested not in claimed:
                entity_id = requested
            else:
                entity_id = self._best_automatic_match(observation, entities, claimed)
            if entity_id is None:
                pending.append(observation)
                continue
            claimed.add(entity_id)
            ref_map[observation.observation_id] = entity_id
            ref_map[entity_id] = entity_id
            resolved.append((observation, entity_id))

        reserved = set(entities) | claimed
        pending.sort(key=lambda item: (item.type, _center(item.bbox_yxyx)[1]))
        for observation in pending:
            entity_id = self._next_id(observation.type, reserved)
            reserved.add(entity_id)
            claimed.add(entity_id)
            ref_map[observation.observation_id] = entity_id
            ref_map[entity_id] = entity_id
            resolved.append((observation, entity_id))
        return resolved, ref_map

    @staticmethod
    def _event_key(event: MemoryEvent) -> tuple[int, str, str, str | None]:
        return event.frame, event.type, event.subject_id, event.object_id

    def _append_event(self, events: list[MemoryEvent], event: MemoryEvent) -> None:
        if self._event_key(event) not in {self._event_key(item) for item in events}:
            events.append(event)

    def _apply_vlm_event(
        self,
        event: EventObservation,
        ref_map: dict[str, str],
        entities: dict[str, EntityState],
        events: list[MemoryEvent],
        relations: dict[tuple[str, str, str], MemoryRelation],
    ) -> None:
        subject_id = ref_map.get(event.subject_ref, event.subject_ref)
        object_id = ref_map.get(event.object_ref, event.object_ref) if event.object_ref else None
        if subject_id not in entities or (object_id is not None and object_id not in entities):
            return
        self._append_event(
            events,
            MemoryEvent(
                frame=event.evidence_frame,
                type=event.type,
                subject_id=subject_id,
                object_id=object_id,
                confidence=event.confidence,
                source="vlm",
                detail=event.detail,
            ),
        )
        if event.type == "covered" and object_id is not None:
            key = (subject_id, "covers", object_id)
            relations[key] = MemoryRelation(
                subject_id=subject_id,
                relation="covers",
                object_id=object_id,
                last_evidence_frame=event.evidence_frame,
                confidence=event.confidence,
            )
        elif event.type == "uncovered" and object_id is not None:
            relations.pop((subject_id, "covers", object_id), None)
        elif event.type == "picked":
            entities[subject_id].held = True
        elif event.type == "placed":
            entities[subject_id].held = False
        elif event.type == "pressed":
            entities[subject_id].pressed = True

    def apply(self, perception: PerceptionOutput, current_frame: int) -> MemoryState:
        entities = {item.id: item.model_copy(deep=True) for item in self.state.entities}
        relations = {
            (item.subject_id, item.relation, item.object_id): item.model_copy(deep=True)
            for item in self.state.relations
        }
        events = [item.model_copy(deep=True) for item in self.state.events]

        for entity in entities.values():
            entity.visibility = "unobserved"
            entity.held = False

        resolved, ref_map = self._resolve_observations(perception.observed_objects, entities)
        for observation, entity_id in resolved:
            previous = entities.get(entity_id)
            if previous is None:
                entities[entity_id] = EntityState(
                    id=entity_id,
                    type=observation.type,
                    bbox_yxyx=copy.deepcopy(observation.bbox_yxyx),
                    color=observation.color,
                    visibility="visible",
                    pressed=observation.pressed,
                    held=observation.held,
                    first_seen_frame=observation.evidence_frame,
                    last_seen_frame=observation.evidence_frame,
                    confidence=observation.confidence,
                )
                self._append_event(
                    events,
                    MemoryEvent(
                        frame=observation.evidence_frame,
                        type="appeared",
                        subject_id=entity_id,
                        object_id=None,
                        confidence=observation.confidence,
                        source="reducer",
                        detail="First direct observation.",
                    ),
                )
                continue

            if _distance(previous.bbox_yxyx, observation.bbox_yxyx) >= MOVE_THRESHOLD_PIXELS:
                self._append_event(
                    events,
                    MemoryEvent(
                        frame=observation.evidence_frame,
                        type="moved",
                        subject_id=entity_id,
                        object_id=None,
                        confidence=observation.confidence,
                        source="reducer",
                        detail="Bounding-box center changed by at least 10 pixels.",
                    ),
                )
            previous.bbox_yxyx = copy.deepcopy(observation.bbox_yxyx)
            previous.color = observation.color or previous.color
            previous.visibility = "visible"
            previous.pressed = observation.pressed if observation.pressed is not None else previous.pressed
            previous.held = observation.held
            previous.last_seen_frame = observation.evidence_frame
            previous.confidence = observation.confidence

        for relation in perception.observed_relations:
            subject_id = ref_map.get(relation.subject_ref, relation.subject_ref)
            object_id = ref_map.get(relation.object_ref, relation.object_ref)
            if subject_id not in entities or object_id not in entities:
                continue
            key = (subject_id, relation.relation, object_id)
            relations[key] = MemoryRelation(
                subject_id=subject_id,
                relation=relation.relation,
                object_id=object_id,
                last_evidence_frame=relation.evidence_frame,
                confidence=relation.confidence,
            )

        for event in perception.observed_events:
            self._apply_vlm_event(event, ref_map, entities, events, relations)

        holding = next((item.id for item in entities.values() if item.held), None)
        self.state = MemoryState(
            current_frame=current_frame,
            entities=sorted(entities.values(), key=lambda item: item.id),
            relations=sorted(relations.values(), key=lambda item: (item.subject_id, item.object_id)),
            events=events[-RECENT_EVENT_LIMIT:],
            robot={"holding": holding},
            uncertainties=perception.uncertainties,
        )
        return self.state.model_copy(deep=True)
