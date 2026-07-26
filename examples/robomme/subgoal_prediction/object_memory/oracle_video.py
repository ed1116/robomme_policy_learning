from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import re

import cv2
import numpy as np

from .schemas import OracleBoundary
from .schemas import Stage

CALL_INTERVAL = 16
FRAME_OFFSETS = (-12, -8, -4, 0)
SUBGOAL_CHANGE_THRESHOLD = 4.0
VIDEO_PATTERN = re.compile(
    r"ButtonUnmaskSwap_ep(?P<episode>\d+)_(?P<outcome>success|fail|timeout)_"
    r"(?P<goal>.+)_(?P<difficulty>easy|medium|hard)\.mp4"
)


@dataclass(frozen=True)
class EpisodeSpec:
    path: Path
    episode_id: int
    outcome: str
    task_goal: str
    difficulty: str


@dataclass
class LoadedEpisode:
    spec: EpisodeSpec
    frame_count: int
    fps: float
    oracle_transition_frames: list[int]
    oracle_stages: list[Stage]
    boundaries: list[OracleBoundary]
    front_frames: dict[int, np.ndarray]

    def window(self, call_frame: int) -> tuple[list[int], list[np.ndarray]]:
        indices = [max(0, call_frame + offset) for offset in FRAME_OFFSETS]
        return indices, [self.front_frames[index] for index in indices]


def parse_episode(path: Path) -> EpisodeSpec:
    match = VIDEO_PATTERN.fullmatch(path.name)
    if match is None:
        raise ValueError(f"Unexpected ButtonUnmaskSwap video filename: {path.name}")
    return EpisodeSpec(
        path=path,
        episode_id=int(match.group("episode")),
        outcome=match.group("outcome"),
        task_goal=match.group("goal"),
        difficulty=match.group("difficulty"),
    )


def select_successive_successes(video_dir: Path, count: int = 5) -> list[EpisodeSpec]:
    episodes = sorted(
        (parse_episode(path) for path in video_dir.glob("ButtonUnmaskSwap_ep*.mp4")),
        key=lambda item: item.episode_id,
    )
    successes = [item for item in episodes if item.outcome == "success"]
    for start in range(len(successes) - count + 1):
        window = successes[start : start + count]
        expected = list(range(window[0].episode_id, window[0].episode_id + count))
        if [item.episode_id for item in window] == expected:
            return window
    raise RuntimeError(f"No block of {count} successive successful ButtonUnmaskSwap oracle rollouts")


def stages_from_goal(task_goal: str) -> list[Stage]:
    colors = re.findall(r"container hiding the (red|green|blue) cube", task_goal.lower())
    if not colors:
        raise ValueError(f"Could not parse target colors from task goal: {task_goal}")
    stages: list[Stage] = [
        "press_first_button",
        "press_second_button",
        "pick_first_target_container",
    ]
    if len(colors) == 2:
        stages.extend(["put_down_container", "pick_second_target_container"])
    if len(colors) > 2:
        raise ValueError(f"ButtonUnmaskSwap supports at most two target colors: {task_goal}")
    return stages


def target_colors(task_goal: str) -> list[str]:
    return re.findall(r"container hiding the (red|green|blue) cube", task_goal.lower())


def _subgoal_band(frame: np.ndarray) -> np.ndarray:
    camera_top = frame.shape[0] - 256
    return frame[max(0, camera_top - 64) : camera_top]


def _front_view(frame: np.ndarray) -> np.ndarray:
    camera_top = frame.shape[0] - 256
    return cv2.cvtColor(frame[camera_top : camera_top + 256, :256], cv2.COLOR_BGR2RGB)


def _oracle_boundaries(
    frame_count: int,
    transition_frames: list[int],
    stages: list[Stage],
) -> list[OracleBoundary]:
    boundaries: list[OracleBoundary] = []
    for call_frame in range(0, frame_count, CALL_INTERVAL):
        effective_frame = max(call_frame, transition_frames[0])
        stage_index = max(
            index for index, transition in enumerate(transition_frames) if transition <= effective_frame
        )
        previous_call = call_frame - CALL_INTERVAL
        changes = [
            transition
            for transition in transition_frames[1:]
            if previous_call < transition <= call_frame
        ]
        boundaries.append(
            OracleBoundary(
                call_frame=call_frame,
                oracle_stage=stages[stage_index],
                oracle_subgoal_completed=bool(changes),
                oracle_change_frame=changes[-1] if changes else None,
            )
        )
    return boundaries


def load_episode(spec: EpisodeSpec) -> LoadedEpisode:
    capture = cv2.VideoCapture(str(spec.path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {spec.path}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = float(capture.get(cv2.CAP_PROP_FPS))
    call_frames = list(range(0, frame_count, CALL_INTERVAL))
    needed = {
        max(0, call_frame + offset)
        for call_frame in call_frames
        for offset in FRAME_OFFSETS
    }

    front_frames: dict[int, np.ndarray] = {}
    transition_frames: list[int] = []
    previous_band: np.ndarray | None = None
    frame_index = 0
    while True:
        ok, frame = capture.read()
        if not ok:
            break
        band = _subgoal_band(frame)
        if previous_band is not None:
            difference = float(np.mean(cv2.absdiff(band, previous_band)))
            if difference >= SUBGOAL_CHANGE_THRESHOLD:
                transition_frames.append(frame_index)
        previous_band = band
        if frame_index in needed:
            front_frames[frame_index] = _front_view(frame)
        frame_index += 1
    capture.release()

    if frame_index != frame_count:
        frame_count = frame_index
    stages = stages_from_goal(spec.task_goal)
    if len(transition_frames) != len(stages):
        raise RuntimeError(
            f"Detected {len(transition_frames)} oracle subgoal segments but expected "
            f"{len(stages)} for episode {spec.episode_id}: {transition_frames}"
        )
    missing = sorted(needed - set(front_frames))
    if missing:
        raise RuntimeError(f"Video extraction missed frame indices: {missing}")
    return LoadedEpisode(
        spec=spec,
        frame_count=frame_count,
        fps=fps,
        oracle_transition_frames=transition_frames,
        oracle_stages=stages,
        boundaries=_oracle_boundaries(frame_count, transition_frames, stages),
        front_frames=front_frames,
    )
