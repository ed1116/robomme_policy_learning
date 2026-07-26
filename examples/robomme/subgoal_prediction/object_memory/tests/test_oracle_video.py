from pathlib import Path

from subgoal_prediction.object_memory.oracle_video import load_episode
from subgoal_prediction.object_memory.oracle_video import parse_episode
from subgoal_prediction.object_memory.oracle_video import select_successive_successes

VIDEO_DIR = Path(
    "runs/evaluation/symbolic-grounded-subgoal/ckpt79999/seed7/oracle/videos"
)


def test_selects_five_successive_successful_episodes():
    episodes = select_successive_successes(VIDEO_DIR, 5)
    assert [item.episode_id for item in episodes] == [3, 4, 5, 6, 7]


def test_detects_oracle_subgoal_transition_frames():
    path = next(VIDEO_DIR.glob("ButtonUnmaskSwap_ep3_success_*.mp4"))
    episode = load_episode(parse_episode(path))
    assert episode.oracle_transition_frames == [1, 113, 289, 481, 513]
    assert episode.oracle_stages == [
        "press_first_button",
        "press_second_button",
        "pick_first_target_container",
        "put_down_container",
        "pick_second_target_container",
    ]
    boundary = next(item for item in episode.boundaries if item.call_frame == 128)
    assert boundary.oracle_subgoal_completed
    assert boundary.oracle_change_frame == 113
    assert boundary.oracle_stage == "press_second_button"
