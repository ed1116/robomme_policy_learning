from subgoal_prediction.object_memory.reducer import MemoryReducer
from subgoal_prediction.object_memory.schemas import PerceptionOutput


def _perception(objects, events=None):
    return PerceptionOutput.model_validate(
        {
            "observed_objects": objects,
            "observed_relations": [],
            "observed_events": events or [],
            "uncertainties": [],
        }
    )


def test_initial_ids_are_assigned_deterministically_left_to_right():
    reducer = MemoryReducer()
    state = reducer.apply(
        _perception(
            [
                {
                    "observation_id": "right",
                    "entity_id": None,
                    "type": "container",
                    "bbox_yxyx": [40, 170, 100, 220],
                    "color": "gray",
                    "pressed": None,
                    "held": False,
                    "evidence_frame": 0,
                    "confidence": 0.9,
                },
                {
                    "observation_id": "left",
                    "entity_id": None,
                    "type": "container",
                    "bbox_yxyx": [40, 20, 100, 70],
                    "color": "gray",
                    "pressed": None,
                    "held": False,
                    "evidence_frame": 0,
                    "confidence": 0.9,
                },
            ]
        ),
        current_frame=0,
    )

    assert [(item.id, item.bbox_yxyx[1]) for item in state.entities] == [
        ("container_a", 20),
        ("container_b", 170),
    ]


def test_vlm_identity_survives_a_position_swap_and_relation_refs_are_resolved():
    reducer = MemoryReducer()
    reducer.apply(
        _perception(
            [
                {
                    "observation_id": "left",
                    "entity_id": None,
                    "type": "container",
                    "bbox_yxyx": [40, 20, 100, 70],
                    "color": "gray",
                    "pressed": None,
                    "held": False,
                    "evidence_frame": 0,
                    "confidence": 0.9,
                },
                {
                    "observation_id": "right",
                    "entity_id": None,
                    "type": "container",
                    "bbox_yxyx": [40, 170, 100, 220],
                    "color": "gray",
                    "pressed": None,
                    "held": False,
                    "evidence_frame": 0,
                    "confidence": 0.9,
                },
            ]
        ),
        current_frame=0,
    )
    state = reducer.apply(
        _perception(
            [
                {
                    "observation_id": "tracked_a",
                    "entity_id": "container_a",
                    "type": "container",
                    "bbox_yxyx": [40, 170, 100, 220],
                    "color": "gray",
                    "pressed": None,
                    "held": False,
                    "evidence_frame": 16,
                    "confidence": 0.8,
                },
                {
                    "observation_id": "cube",
                    "entity_id": None,
                    "type": "cube",
                    "bbox_yxyx": [100, 35, 120, 55],
                    "color": "red",
                    "pressed": None,
                    "held": False,
                    "evidence_frame": 16,
                    "confidence": 0.9,
                },
            ],
            events=[
                {
                    "type": "uncovered",
                    "subject_ref": "tracked_a",
                    "object_ref": "cube",
                    "evidence_frame": 16,
                    "confidence": 0.8,
                    "detail": "The tracked container revealed the red cube.",
                }
            ],
        ),
        current_frame=16,
    )

    container_a = next(item for item in state.entities if item.id == "container_a")
    uncovered = next(item for item in state.events if item.type == "uncovered")
    assert container_a.bbox_yxyx[1] == 170
    assert uncovered.subject_id == "container_a"
    assert uncovered.object_id == "cube_a"
