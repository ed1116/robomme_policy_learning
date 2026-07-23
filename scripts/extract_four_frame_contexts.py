"""Extract chronological four-frame contexts from a RoboMME evaluation video."""

from __future__ import annotations

import argparse
from collections import deque
import json
from pathlib import Path

import cv2


VIEW_HEIGHT = 256
VIEW_WIDTH = 256


def crop_view(frame, view: str):
    """Crop camera pixels from the annotated evaluation-video frame."""
    height, width = frame.shape[:2]
    if view == "full":
        return frame
    if height < VIEW_HEIGHT or width < VIEW_WIDTH:
        raise ValueError(
            f"Video frame is {width}x{height}; expected at least {VIEW_WIDTH}x{VIEW_HEIGHT}."
        )

    camera_row = frame[-VIEW_HEIGHT:]
    if view == "front":
        return camera_row[:, :VIEW_WIDTH]
    if view == "wrist":
        if width < 2 * VIEW_WIDTH:
            raise ValueError(f"Video frame width {width} does not contain a wrist view.")
        return camera_row[:, VIEW_WIDTH : 2 * VIEW_WIDTH]
    if view == "concat":
        if width < 2 * VIEW_WIDTH:
            raise ValueError(f"Video frame width {width} does not contain two camera views.")
        return camera_row[:, : 2 * VIEW_WIDTH]
    raise ValueError(f"Unknown view: {view}")


def extract_contexts(
    video_path: Path,
    output_dir: Path,
    *,
    stride: int,
    view: str,
) -> dict:
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    output_dir.mkdir(parents=True, exist_ok=True)
    sampled_frames: deque[tuple[int, object]] = deque(maxlen=4)
    contexts = []
    frame_index = 0

    try:
        while True:
            ok, frame = capture.read()
            if not ok:
                break

            if frame_index == 0:
                cv2.imwrite(str(output_dir / "initial_frame_0.png"), crop_view(frame, view))
            elif frame_index % stride == 0:
                sampled_frames.append((frame_index, crop_view(frame, view)))

                if len(sampled_frames) == 4:
                    current_index = sampled_frames[-1][0]
                    context_dir = output_dir / f"context_t{current_index:06d}"
                    context_dir.mkdir(exist_ok=True)

                    offsets = (-3 * stride, -2 * stride, -stride, 0)
                    image_records = []
                    for image_number, ((source_index, image), offset) in enumerate(
                        zip(sampled_frames, offsets, strict=True), start=1
                    ):
                        offset_name = "t" if offset == 0 else f"t_minus_{abs(offset)}"
                        image_path = context_dir / f"image_{image_number}_{offset_name}.png"
                        if not cv2.imwrite(str(image_path), image):
                            raise RuntimeError(f"Failed to write image: {image_path}")
                        image_records.append(
                            {
                                "image": image_number,
                                "time": offset_name,
                                "source_frame": source_index,
                                "path": str(image_path.relative_to(output_dir)),
                            }
                        )

                    contexts.append(
                        {
                            "current_frame": current_index,
                            "directory": str(context_dir.relative_to(output_dir)),
                            "images": image_records,
                        }
                    )
                    sampled_frames.clear()

            frame_index += 1
    finally:
        capture.release()

    if frame_index == 0:
        raise RuntimeError(f"No frames could be decoded from: {video_path}")
    if not contexts:
        raise RuntimeError(
            f"The video has {frame_index} frames, fewer than the {4 * stride + 1} "
            "needed for an initial frame and one four-frame context."
        )

    manifest = {
        "source_video": str(video_path.resolve()),
        "source_frame_count": frame_index,
        "view": view,
        "stride": stride,
        "initial_frame": "initial_frame_0.png",
        "context_offsets": [-3 * stride, -2 * stride, -stride, 0],
        "context_count": len(contexts),
        "contexts": contexts,
    }
    with (output_dir / "manifest.json").open("w") as file:
        json.dump(manifest, file, indent=2)
    return manifest


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create an initial frame-0 context, then four-image contexts sampled at "
            "t-12, t-8, t-4, and t for t=16,32,48,... by default."
        )
    )
    parser.add_argument("video_path", type=Path, help="Path to a RoboMME evaluation MP4")
    parser.add_argument(
        "--output-dir",
        type=Path,
        help="Output directory (default: <video_stem>_four_frame_contexts)",
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=4,
        help="Number of video frames between supplied images (default: 4)",
    )
    parser.add_argument(
        "--view",
        choices=("front", "wrist", "concat", "full"),
        default="front",
        help="Image region to extract; front matches the current QwenVL input (default: front)",
    )
    args = parser.parse_args()
    if args.stride <= 0:
        parser.error("--stride must be greater than zero")
    if not args.video_path.is_file():
        parser.error(f"video does not exist: {args.video_path}")
    if args.output_dir is None:
        args.output_dir = args.video_path.with_name(
            f"{args.video_path.stem}_four_frame_contexts"
        )
    return args


def main() -> None:
    args = parse_args()
    manifest = extract_contexts(
        args.video_path,
        args.output_dir,
        stride=args.stride,
        view=args.view,
    )
    print(
        f"Created {manifest['context_count']} contexts in {args.output_dir} "
        f"using the {manifest['view']} view."
    )


if __name__ == "__main__":
    main()
