from pathlib import Path

from eval_object_memory_offline import _canonical_predicted_stage
from subgoal_prediction.object_memory.oracle_video import load_episode
from subgoal_prediction.object_memory.oracle_video import parse_episode
from subgoal_prediction.object_memory.oracle_video import select_successive_successes

VIDEO_DIR = Path(
    "runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/oracle/videos"
)


def test_selects_five_successive_successful_episodes():
    episodes = select_successive_successes(VIDEO_DIR, 5)
    assert [item.episode_id for item in episodes] == [0, 1, 2, 3, 4]


def test_detects_oracle_subgoal_transition_frames():
    path = next(VIDEO_DIR.glob("ButtonUnmaskSwap_ep3_success_*.mp4"))
    episode = load_episode(parse_episode(path))
    assert episode.oracle_text_transition_frames == [1, 113, 209, 417, 449]
    assert episode.oracle_completion_frames == [0, 112, 208, 416, 448]
    assert episode.oracle_stages == [
        "press_first_button",
        "press_second_button",
        "pick_first_target_container",
        "put_down_container",
        "pick_second_target_container",
    ]
    boundary = next(item for item in episode.boundaries if item.call_frame == 112)
    assert boundary.oracle_subgoal_completed
    assert boundary.oracle_change_frame == 112
    assert boundary.oracle_stage == "press_second_button"
    indices, _ = episode.window(128)
    assert indices == [116, 120, 124, 128]


def test_free_form_subgoals_are_canonicalized_only_for_evaluation():
    assert (
        _canonical_predicted_stage(
            {
                "stage": "pick up the container that hides the green cube",
                "target_color": "green",
            },
            ["green", "blue"],
        )
        == "pick_first_target_container"
    )
