import json
from types import SimpleNamespace

import numpy as np
from subgoal_prediction.object_memory.planner import SplitObjectMemoryPlanner
from subgoal_prediction.object_memory.schemas import DecisionOutput
from subgoal_prediction.object_memory.schemas import InitialPerceptionOutput
from subgoal_prediction.object_memory.schemas import PerceptionOutput


class FakeBackend:
    def __init__(self):
        self.calls = []

    def infer(self, *, image_paths, schema, system_prompt, **_):
        self.calls.append(
            {
                "has_images": bool(image_paths),
                "schema": schema,
                "system_prompt": system_prompt,
            }
        )
        if image_paths:
            if schema is InitialPerceptionOutput:
                parsed = InitialPerceptionOutput.model_validate(
                    {
                        "observed_objects": [
                            {
                                "observation_id": "visible_button",
                                "entity_id": None,
                                "type": "button",
                                "bbox_xyxy_norm1000": [100, 100, 200, 200],
                                "color": "gray",
                                "pressed": False,
                                "held": False,
                                "evidence_frame": 0,
                            }
                        ],
                        "observed_visibility_changes": [],
                        "uncertainties": [],
                    }
                )
            else:
                parsed = PerceptionOutput.model_validate(
                    {
                        "observed_objects": [],
                        "observed_visibility_changes": [],
                        "uncertainties": [],
                    }
                )
        else:
            parsed = DecisionOutput.model_validate(
                {
                    "subgoal_completed": False,
                    "completion_evidence": "No active subgoal exists on initialization.",
                    "next_subgoal": {
                        "stage": "press the first button",
                        "target_entity_id": "button_a",
                        "target_color": None,
                    },
                }
            )
        return SimpleNamespace(parsed=parsed, attempts=[{"parsed_output": parsed.model_dump(mode="json")}])


def test_each_cycle_writes_four_explicit_stage_outputs(tmp_path):
    planner = SplitObjectMemoryPlanner(FakeBackend())
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    planner.run_cycle(
        task_goal="Press the button.",
        call_frame=0,
        frame_indices=[0, 0, 0, 0],
        frames=[frame, frame, frame, frame],
        call_dir=tmp_path,
    )

    perception = json.loads((tmp_path / "perception_output.json").read_text())
    reducer = json.loads((tmp_path / "reducer_output.json").read_text())
    decision = json.loads((tmp_path / "decision_output.json").read_text())
    grounded = json.loads((tmp_path / "grounded_subgoal_output.json").read_text())

    assert perception["observed_objects"][0]["observation_id"] == "visible_button"
    assert reducer["entities"][0]["id"] == "button_a"
    assert reducer["entities"][0] == {"id": "button_a", "type": "button"}
    assert reducer["states"][0]["bbox_yxyx"] == [26, 26, 51, 51]
    assert decision["next_subgoal"]["target_entity_id"] == "button_a"
    assert decision["next_subgoal"]["stage"] == "press the first button"
    assert grounded["target_entity_id"] == "button_a"
    assert grounded["point_yx"] == [38, 38]


def test_subgoal_stage_is_not_a_closed_enum():
    decision = DecisionOutput.model_validate(
        {
            "subgoal_completed": False,
            "completion_evidence": "The action is still in progress.",
            "next_subgoal": {
                "stage": "press_button",
                "target_entity_id": "button_a",
                "target_color": None,
            },
        }
    )

    assert decision.next_subgoal.stage == "press_button"


def test_perception_uses_initial_then_delta_only_system_prompt(tmp_path):
    backend = FakeBackend()
    planner = SplitObjectMemoryPlanner(backend)
    frame = np.zeros((256, 256, 3), dtype=np.uint8)

    planner.run_cycle(
        task_goal="Press the button.",
        call_frame=0,
        frame_indices=[0, 0, 0, 0],
        frames=[frame, frame, frame, frame],
        call_dir=tmp_path / "call_t0000",
    )
    planner.run_cycle(
        task_goal="Press the button.",
        call_frame=16,
        frame_indices=[4, 8, 12, 16],
        frames=[frame, frame, frame, frame],
        call_dir=tmp_path / "call_t0016",
    )

    perception_calls = [call for call in backend.calls if call["has_images"]]
    assert [call["schema"] for call in perception_calls] == [
        InitialPerceptionOutput,
        PerceptionOutput,
    ]
    assert perception_calls[0]["system_prompt"].startswith(
        "PERCEPTION INITIALIZATION CALL"
    )
    assert perception_calls[1]["system_prompt"].startswith(
        "PERCEPTION DELTA CALL"
    )
    assert "Existing tracks provide" in (
        perception_calls[1]["system_prompt"]
    )
    assert "yxyx" not in perception_calls[1]["system_prompt"]
    assert "Do not infer or output relations or events" in (
        perception_calls[1]["system_prompt"]
    )
    assert planner._track_table(planner.reducer.state)[0][
        "last_bbox_xyxy_norm1000"
    ] == [102, 102, 199, 199]

    delta = json.loads(
        (tmp_path / "call_t0016" / "perception_output.json").read_text()
    )
    assert delta["observed_objects"] == []
    assert delta["observed_visibility_changes"] == []
