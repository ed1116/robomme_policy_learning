from pydantic import ValidationError
import pytest
from subgoal_prediction.object_memory.reducer import MemoryReducer
from subgoal_prediction.object_memory.schemas import ObjectObservation
from subgoal_prediction.object_memory.schemas import PerceptionOutput


def _perception(objects=None, visibility_changes=None, uncertainties=None):
    return PerceptionOutput.model_validate(
        {
            "observed_objects": objects or [],
            "observed_visibility_changes": visibility_changes or [],
            "uncertainties": uncertainties or [],
        }
    )


def _object(
    observation_id,
    entity_type,
    bbox,
    *,
    entity_id=None,
    evidence_frame=0,
    color=None,
    pressed=None,
    held=None,
):
    return {
        "observation_id": observation_id,
        "entity_id": entity_id,
        "type": entity_type,
        "bbox_xyxy_norm1000": bbox,
        "color": color,
        "pressed": pressed,
        "held": held,
        "evidence_frame": evidence_frame,
    }


def test_initial_ids_are_assigned_deterministically_left_to_right():
    reducer = MemoryReducer()
    state = reducer.apply(
        _perception(
            [
                _object("right", "container", [664, 156, 859, 391]),
                _object("left", "container", [78, 156, 273, 391]),
            ]
        ),
        current_frame=0,
    )

    assert [(item.id, item.bbox_yxyx[1]) for item in state.states] == [
        ("container_a", 20),
        ("container_b", 170),
    ]
    assert [event.type for event in state.events] == ["appeared", "appeared"]
    assert all(event.source == "reducer" for event in state.events)


def test_explicit_identity_survives_a_position_swap():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object("left", "container", [78, 156, 273, 391]),
                _object("right", "container", [664, 156, 859, 391]),
            ]
        ),
        current_frame=0,
    )

    state = reducer.apply(
        _perception(
            [
                _object(
                    "tracked_a",
                    "container",
                    [664, 156, 859, 391],
                    entity_id="container_a",
                    evidence_frame=16,
                )
            ]
        ),
        current_frame=16,
    )

    container_a = next(item for item in state.states if item.id == "container_a")
    assert container_a.bbox_yxyx[1] == 170
    assert any(
        event.type == "moved" and event.subject_id == "container_a"
        for event in state.events
    )


def test_color_change_does_not_create_a_new_entity():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object(
                    "first",
                    "cube",
                    [100, 100, 200, 200],
                    color="red",
                )
            ]
        ),
        current_frame=0,
    )

    state = reducer.apply(
        _perception(
            [
                _object(
                    "second",
                    "cube",
                    [110, 100, 210, 200],
                    color="blue",
                    evidence_frame=16,
                )
            ]
        ),
        current_frame=16,
    )

    assert [item.id for item in state.entities] == ["cube_a"]
    assert state.states[0].color == "blue"


def _initialize_three_cubes(reducer):
    reducer.apply(
        _perception(
            [
                _object(
                    "green",
                    "cube",
                    [400, 300, 450, 350],
                    color="green",
                ),
                _object(
                    "blue",
                    "cube",
                    [650, 450, 700, 500],
                    color="blue",
                ),
                _object(
                    "red",
                    "cube",
                    [650, 300, 700, 350],
                    color="red",
                ),
            ]
        ),
        current_frame=0,
    )


def _cover_three_cubes(reducer):
    return reducer.apply(
        _perception(
            [
                _object(
                    "empty_container",
                    "container",
                    [250, 450, 350, 550],
                    evidence_frame=32,
                ),
                _object(
                    "green_container",
                    "container",
                    [380, 280, 470, 370],
                    evidence_frame=32,
                ),
                _object(
                    "blue_container",
                    "container",
                    [630, 430, 720, 520],
                    evidence_frame=32,
                ),
                _object(
                    "red_container",
                    "container",
                    [630, 280, 720, 370],
                    evidence_frame=32,
                ),
            ],
            visibility_changes=[
                {
                    "entity_id": "cube_a",
                    "visibility": "occluded",
                    "evidence_frame": 32,
                },
                {
                    "entity_id": "cube_b",
                    "visibility": "occluded",
                    "evidence_frame": 32,
                },
                {
                    "entity_id": "cube_c",
                    "visibility": "occluded",
                    "evidence_frame": 32,
                },
            ],
        ),
        current_frame=32,
    )


def test_strict_center_containment_creates_one_to_one_cover_relations():
    reducer = MemoryReducer()
    _initialize_three_cubes(reducer)

    state = _cover_three_cubes(reducer)

    assert {
        (relation.subject_id, relation.object_id)
        for relation in state.relations
    } == {
        ("container_b", "cube_a"),
        ("container_c", "cube_b"),
        ("container_d", "cube_c"),
    }
    assert all(
        next(item for item in state.states if item.id == cube_id).visibility
        == "occluded"
        for cube_id in ("cube_a", "cube_b", "cube_c")
    )
    covered = [event for event in state.events if event.type == "covered"]
    assert {
        (event.subject_id, event.object_id)
        for event in covered
    } == {
        ("cube_a", "container_b"),
        ("cube_b", "container_c"),
        ("cube_c", "container_d"),
    }
    assert not any(
        relation.subject_id == "container_a"
        for relation in state.relations
    )


def test_nearby_container_without_center_containment_does_not_cover():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object(
                    "cube",
                    "cube",
                    [100, 100, 200, 200],
                    color="green",
                )
            ]
        ),
        current_frame=0,
    )

    state = reducer.apply(
        _perception(
            [
                _object(
                    "near_container",
                    "container",
                    [190, 100, 290, 200],
                    evidence_frame=16,
                )
            ],
            visibility_changes=[
                {
                    "entity_id": "cube_a",
                    "visibility": "occluded",
                    "evidence_frame": 16,
                }
            ],
        ),
        current_frame=16,
    )

    cube = next(item for item in state.states if item.id == "cube_a")
    assert cube.visibility == "unobserved"
    assert state.relations == []
    assert not any(event.type == "covered" for event in state.events)
    assert "candidates=[]" in state.uncertainties[-1]


def test_ambiguous_cover_candidates_are_rejected():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object(
                    "cube",
                    "cube",
                    [100, 100, 200, 200],
                    color="green",
                )
            ]
        ),
        current_frame=0,
    )

    state = reducer.apply(
        _perception(
            [
                _object(
                    "container_1",
                    "container",
                    [80, 80, 220, 220],
                    evidence_frame=16,
                ),
                _object(
                    "container_2",
                    "container",
                    [100, 100, 230, 230],
                    evidence_frame=16,
                ),
            ],
            visibility_changes=[
                {
                    "entity_id": "cube_a",
                    "visibility": "occluded",
                    "evidence_frame": 16,
                }
            ],
        ),
        current_frame=16,
    )

    assert state.relations == []
    assert not any(event.type == "covered" for event in state.events)
    assert "container_a" in state.uncertainties[-1]
    assert "container_b" in state.uncertainties[-1]


def test_reappearing_cube_removes_cover_and_creates_uncovered_event():
    reducer = MemoryReducer()
    _initialize_three_cubes(reducer)
    _cover_three_cubes(reducer)

    state = reducer.apply(
        _perception(
            [
                _object(
                    "visible_green",
                    "cube",
                    [400, 300, 450, 350],
                    entity_id="cube_a",
                    evidence_frame=48,
                    color="green",
                )
            ]
        ),
        current_frame=48,
    )

    assert not any(
        relation.object_id == "cube_a"
        for relation in state.relations
    )
    assert any(
        event.type == "uncovered"
        and event.subject_id == "cube_a"
        and event.object_id == "container_b"
        for event in state.events
    )
    cube = next(item for item in state.states if item.id == "cube_a")
    assert cube.visibility == "visible"


def test_pressed_state_transition_synthesizes_pressed_event():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object(
                    "button",
                    "button",
                    [100, 100, 200, 200],
                    pressed=False,
                )
            ]
        ),
        current_frame=0,
    )

    state = reducer.apply(
        _perception(
            [
                _object(
                    "pressed_button",
                    "button",
                    [100, 100, 200, 200],
                    entity_id="button_a",
                    evidence_frame=112,
                    pressed=True,
                )
            ]
        ),
        current_frame=112,
    )

    pressed = [event for event in state.events if event.type == "pressed"]
    assert len(pressed) == 1
    assert pressed[0].subject_id == "button_a"
    assert pressed[0].source == "reducer"


def test_held_transitions_synthesize_picked_and_placed_events():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object(
                    "container",
                    "container",
                    [100, 100, 200, 200],
                    held=False,
                )
            ]
        ),
        current_frame=0,
    )

    picked = reducer.apply(
        _perception(
            [
                _object(
                    "held_container",
                    "container",
                    [100, 100, 200, 200],
                    entity_id="container_a",
                    evidence_frame=16,
                    held=True,
                )
            ]
        ),
        current_frame=16,
    )
    assert picked.robot.holding == "container_a"
    assert any(event.type == "picked" for event in picked.events)

    placed = reducer.apply(
        _perception(
            [
                _object(
                    "released_container",
                    "container",
                    [100, 100, 200, 200],
                    entity_id="container_a",
                    evidence_frame=32,
                    held=False,
                )
            ]
        ),
        current_frame=32,
    )
    assert placed.robot.holding is None
    assert any(event.type == "placed" for event in placed.events)


def test_empty_delta_preserves_unchanged_visible_object():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                _object(
                    "button",
                    "button",
                    [100, 100, 200, 200],
                    pressed=False,
                )
            ]
        ),
        current_frame=0,
    )

    state = reducer.apply(_perception(), current_frame=16)

    assert len(state.entities) == 1
    assert state.states[0].visibility == "visible"
    assert state.states[0].motion == "stationary"
    assert state.states[0].last_seen_frame == 0


def test_bbox_deadband_uses_each_xyxy_coordinate_in_normalized_space():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [_object("initial", "button", [100, 100, 200, 200])]
        ),
        current_frame=0,
    )
    initial_bbox = reducer.state.states[0].bbox_yxyx.copy()

    unchanged = reducer.apply(
        _perception(
            [
                _object(
                    "jitter",
                    "button",
                    [121, 102, 218, 199],
                    entity_id="button_a",
                    evidence_frame=16,
                )
            ]
        ),
        current_frame=16,
    )
    assert unchanged.states[0].motion == "stationary"
    assert unchanged.states[0].bbox_yxyx == initial_bbox
    assert not any(event.type == "moved" for event in unchanged.events)

    moved = reducer.apply(
        _perception(
            [
                _object(
                    "moved",
                    "button",
                    [122, 102, 219, 199],
                    entity_id="button_a",
                    evidence_frame=32,
                )
            ]
        ),
        current_frame=32,
    )
    assert moved.states[0].motion == "moving"
    assert moved.states[0].bbox_yxyx[1] - initial_bbox[1] == 5
    assert any(event.type == "moved" for event in moved.events)


def test_perception_schema_rejects_vlm_relations_and_events():
    with pytest.raises(ValidationError):
        PerceptionOutput.model_validate(
            {
                "observed_objects": [],
                "observed_visibility_changes": [],
                "observed_relations": [],
                "observed_events": [],
                "uncertainties": [],
            }
        )


def test_integer_observation_ids_are_normalized_before_validation():
    perception = PerceptionOutput.model_validate(
        {
            "observed_objects": [
                _object(1, "container", [100, 100, 200, 200])
            ],
            "observed_visibility_changes": [],
            "uncertainties": [],
        }
    )

    assert perception.observed_objects[0].observation_id == "observation_1"


def test_reducer_rejects_duplicate_observation_ids_from_unvalidated_input():
    duplicate = ObjectObservation.model_construct(
        observation_id="observation_1",
        entity_id=None,
        type="button",
        bbox_xyxy_norm1000=[100, 100, 200, 200],
        evidence_frame=0,
    )
    perception = PerceptionOutput.model_construct(
        observed_objects=[duplicate, duplicate.model_copy(deep=True)],
        observed_visibility_changes=[],
        uncertainties=[],
    )

    with pytest.raises(
        ValueError,
        match="observation_id must be unique within one response",
    ):
        MemoryReducer().apply(perception, current_frame=0)
