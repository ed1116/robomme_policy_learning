#!/usr/bin/env python3
"""Run SAM 3/3.1 text-prompted video segmentation and save visual/text output."""

from __future__ import annotations

import argparse
import json
import subprocess
import tempfile
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

PROMPT_COLORS_BGR = {
    "cube": (80, 220, 80),
    "container": (230, 190, 50),
    "robot arm": (210, 80, 220),
    "button": (40, 150, 255),
}
BUTTON_UNMASK_SWAP_QUERY_SPECS = [
    ("cube", "colored cube"),
    ("button", "white square button"),
    ("robot arm", "robot"),
    ("robot arm", "robot arm"),
    ("robot arm", "robot gripper"),
    ("robot arm", "gripper"),
    ("robot arm", "robotic gripper"),
    ("robot arm", "robot end effector"),
]
CLASS_PRIORITY = {"robot arm": 0, "cube": 1, "container": 2, "button": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--video", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--model-version", choices=("sam3.1", "sam3"), required=True
    )
    parser.add_argument(
        "--prompts",
        nargs="+",
        default=["cube", "container", "robot arm", "button"],
    )
    parser.add_argument("--prompt-frame", type=int, default=0)
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--mask-alpha", type=float, default=0.42)
    parser.add_argument(
        "--button-unmask-swap-layout",
        action="store_true",
        help=(
            "Use audited ButtonUnmaskSwap seed queries, split upper/lower white "
            "objects into buttons/containers, and suppress duplicate masks."
        ),
    )
    parser.add_argument(
        "--keep-mp4v",
        action="store_true",
        help="Skip the final H.264 re-encode.",
    )
    return parser.parse_args()


def load_video(path: Path) -> tuple[list[np.ndarray], float]:
    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {path}")
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    frames: list[np.ndarray] = []
    while True:
        ok, frame = cap.read()
        if not ok:
            break
        frames.append(frame)
    cap.release()
    if not frames:
        raise RuntimeError(f"Video contains no readable frames: {path}")
    return frames, float(fps)


def to_numpy(value: Any) -> np.ndarray:
    if isinstance(value, torch.Tensor):
        return value.detach().cpu().numpy()
    return np.asarray(value)


def mask_record(
    *,
    prompt: str,
    frame_index: int,
    track_id: int,
    score: float | None,
    mask: np.ndarray,
) -> dict[str, Any] | None:
    ys, xs = np.nonzero(mask)
    if len(xs) == 0:
        return None
    height, width = mask.shape
    x1, x2 = int(xs.min()), int(xs.max()) + 1
    y1, y2 = int(ys.min()), int(ys.max()) + 1
    xyxy_norm1000 = [
        round(1000 * x1 / width),
        round(1000 * y1 / height),
        round(1000 * x2 / width),
        round(1000 * y2 / height),
    ]
    yxyx_256 = [
        round(256 * y1 / height),
        round(256 * x1 / width),
        round(256 * y2 / height),
        round(256 * x2 / width),
    ]
    return {
        "frame_index": frame_index,
        "prompt": prompt,
        "track_id": track_id,
        "score": None if score is None else round(score, 6),
        "bbox_xyxy_pixels": [x1, y1, x2, y2],
        "bbox_xyxy_norm1000": xyxy_norm1000,
        "bbox_yxyx_256": yxyx_256,
        "centroid_xy_pixels": [round(float(xs.mean()), 2), round(float(ys.mean()), 2)],
        "mask_area_pixels": int(mask.sum()),
    }


def annotate_output(
    *,
    track_prompts: dict[int, str],
    track_seed_metadata: dict[int, dict[str, Any]],
    frame_index: int,
    output: dict[str, Any],
    frame: np.ndarray,
    alpha: float,
) -> list[dict[str, Any]]:
    masks = to_numpy(output.get("out_binary_masks", np.empty((0,))))
    obj_ids = to_numpy(output.get("out_obj_ids", np.empty((0,), dtype=np.int64)))
    probs_raw = output.get("out_probs")
    probs = None if probs_raw is None else to_numpy(probs_raw)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    detections: list[dict[str, Any]] = []
    for index, (obj_id, raw_mask) in enumerate(zip(obj_ids, masks)):
        track_id = int(obj_id)
        prompt = track_prompts.get(track_id, "unknown")
        color = np.asarray(
            PROMPT_COLORS_BGR.get(prompt, (255, 255, 255)), dtype=np.uint8
        )
        mask = np.asarray(raw_mask).astype(bool)
        if mask.shape != frame.shape[:2]:
            mask = cv2.resize(
                mask.astype(np.uint8),
                (frame.shape[1], frame.shape[0]),
                interpolation=cv2.INTER_NEAREST,
            ).astype(bool)
        score = None
        if probs is not None and index < len(probs):
            score = float(np.asarray(probs[index]).reshape(-1)[0])
        record = mask_record(
            prompt=prompt,
            frame_index=frame_index,
            track_id=track_id,
            score=score,
            mask=mask,
        )
        if record is None:
            continue
        seed_metadata = track_seed_metadata.get(track_id, {})
        record["score_source"] = "SAM 3.1 propagation out_probs"
        record["seed_semantic_score"] = seed_metadata.get("seed_semantic_score")
        record["seed_semantic_query"] = seed_metadata.get("semantic_query")
        if "classification_rule" in seed_metadata:
            record["classification_rule"] = seed_metadata["classification_rule"]
        frame[mask] = (
            (1.0 - alpha) * frame[mask] + alpha * color
        ).astype(np.uint8)
        x1, y1, x2, y2 = record["bbox_xyxy_pixels"]
        label = f"{prompt}:{track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 1)
        cv2.putText(
            frame,
            label,
            (x1, max(12, y1 - 4)),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.38,
            color.tolist(),
            1,
            cv2.LINE_AA,
        )
        detections.append(record)
    return detections


def build_predictor(args: argparse.Namespace):
    if args.model_version == "sam3.1" and torch.cuda.get_device_capability()[0] < 8:
        import sam3.model.decoder as decoder_module
        from torch.nn.attention import SDPBackend
        from torch.nn.attention import sdpa_kernel as torch_sdpa_kernel

        decoder_module.sdpa_kernel = lambda _backends: torch_sdpa_kernel(
            SDPBackend.MATH
        )

    from sam3.model_builder import build_sam3_predictor

    common = {
        "checkpoint_path": str(args.checkpoint),
        "version": args.model_version,
        "compile": False,
        "async_loading_frames": False,
    }
    if args.model_version == "sam3.1":
        common.update(
            {
                "use_fa3": False,
                "use_rope_real": False,
                "warm_up": False,
                "default_output_prob_thresh": args.score_threshold,
            }
        )
    predictor = build_sam3_predictor(**common)
    if args.model_version == "sam3.1":
        predictor.model.batched_grounding_batch_size = 4
        original_init_state = predictor.model.init_state

        def init_state_compat(*init_args, offload_state_to_cpu=False, **init_kwargs):
            del offload_state_to_cpu
            return original_init_state(*init_args, **init_kwargs)

        predictor.model.init_state = init_state_compat
    return predictor


def collect_seed_detections(
    *,
    predictor: Any,
    session_id: str,
    label: str,
    query: str,
    prompt_frame: int,
    score_threshold: float,
) -> list[dict[str, Any]]:
    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": prompt_frame,
            "text": query,
            "output_prob_thresh": score_threshold,
        }
    )
    output = response["outputs"]
    masks = to_numpy(output.get("out_binary_masks", np.empty((0,))))
    obj_ids = to_numpy(output.get("out_obj_ids", np.empty((0,), dtype=np.int64)))
    probs_raw = output.get("out_probs")
    probs = None if probs_raw is None else to_numpy(probs_raw)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    seeds: list[dict[str, Any]] = []
    for index, (source_obj_id, raw_mask) in enumerate(zip(obj_ids, masks)):
        mask = np.asarray(raw_mask).astype(bool)
        ys, xs = np.nonzero(mask)
        if len(xs) == 0:
            continue
        mean_x = float(xs.mean())
        mean_y = float(ys.mean())
        closest = int(np.argmin((xs - mean_x) ** 2 + (ys - mean_y) ** 2))
        score = None
        if probs is not None and index < len(probs):
            score = float(np.asarray(probs[index]).reshape(-1)[0])
        record = mask_record(
            prompt=label,
            frame_index=prompt_frame,
            track_id=int(source_obj_id),
            score=score,
            mask=mask,
        )
        if record is None:
            continue
        record["source_obj_id"] = int(source_obj_id)
        record["semantic_query"] = query
        record["centroid_xy_rel"] = [
            float(xs.mean() / mask.shape[1]),
            float(ys.mean() / mask.shape[0]),
        ]
        record["seed_point_xy_rel"] = [
            float(xs[closest] / mask.shape[1]),
            float(ys[closest] / mask.shape[0]),
        ]
        record["_mask"] = mask
        seeds.append(record)
    return seeds


def resolve_button_unmask_swap_seeds(
    seeds: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    raw_counts_by_query = {
        query: sum(seed["semantic_query"] == query for seed in seeds)
        for _, query in BUTTON_UNMASK_SWAP_QUERY_SPECS
    }
    for seed in seeds:
        if seed["prompt"] == "button" and seed["centroid_xy_rel"][1] >= 0.5:
            seed["prompt"] = "container"
            seed["classification_rule"] = "lower white square movable object"

    ranked_indices = sorted(
        range(len(seeds)),
        key=lambda index: (
            CLASS_PRIORITY[seeds[index]["prompt"]],
            seeds[index]["score"] if seeds[index]["score"] is not None else -1.0,
        ),
        reverse=True,
    )
    retained_indices: list[int] = []
    for index in ranked_indices:
        mask = seeds[index]["_mask"]
        area = int(mask.sum())
        if area == 0:
            continue
        is_duplicate = False
        for retained_index in retained_indices:
            retained_mask = seeds[retained_index]["_mask"]
            intersection = int(np.logical_and(mask, retained_mask).sum())
            smaller_area = min(area, int(retained_mask.sum()))
            if smaller_area > 0 and intersection / smaller_area >= 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            retained_indices.append(index)

    retained_indices.sort()
    resolved = [seeds[index] for index in retained_indices]
    final_counts = {
        label: sum(seed["prompt"] == label for seed in resolved)
        for label in PROMPT_COLORS_BGR
    }
    audit = {
        "enabled": True,
        "raw_instances": len(seeds),
        "raw_instance_counts_by_query": raw_counts_by_query,
        "white_object_classification": {
            "button": "centroid y < 0.5 * image height",
            "container": "centroid y >= 0.5 * image height",
        },
        "duplicate_mask_containment_threshold": 0.8,
        "final_instances": len(resolved),
        "final_instance_counts": final_counts,
    }
    return resolved, audit


def run_joint_tracking(
    *,
    predictor: Any,
    video: Path,
    prompts: list[str],
    prompt_frame: int,
    score_threshold: float,
    annotated_frames: list[np.ndarray],
    mask_alpha: float,
    button_unmask_swap_layout: bool,
) -> tuple[
    dict[int, list[dict[str, Any]]],
    dict[str, dict[str, Any]],
    dict[int, str],
    float,
]:
    started = time.monotonic()
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(video)}
    )
    session_id = response["session_id"]
    try:
        seeds: list[dict[str, Any]] = []
        prompt_summaries: dict[str, dict[str, Any]] = {}
        query_specs = (
            BUTTON_UNMASK_SWAP_QUERY_SPECS
            if button_unmask_swap_layout
            else [(prompt, prompt) for prompt in prompts]
        )
        for label, query in query_specs:
            prompt_seeds = collect_seed_detections(
                predictor=predictor,
                session_id=session_id,
                label=label,
                query=query,
                prompt_frame=prompt_frame,
                score_threshold=score_threshold,
            )
            seeds.extend(prompt_seeds)
            prompt_summaries[query] = {
                "output_class": label,
                "seed_instances": len(prompt_seeds),
                "seed_scores": [seed["score"] for seed in prompt_seeds],
                "seed_boxes_xyxy_norm1000": [
                    seed["bbox_xyxy_norm1000"] for seed in prompt_seeds
                ],
            }

        if button_unmask_swap_layout:
            seeds, postprocess_audit = resolve_button_unmask_swap_seeds(seeds)
        else:
            postprocess_audit = {"enabled": False}
        if not seeds:
            raise RuntimeError("No objects were detected on the prompt frame.")
        if len(seeds) > 16:
            raise RuntimeError(
                f"Detected {len(seeds)} seed instances, exceeding the 16-object "
                "SAM 3.1 multiplex capacity."
            )

        predictor.handle_request(
            {
                "type": "reset_session",
                "session_id": session_id,
            }
        )

        track_prompts: dict[int, str] = {}
        track_seed_metadata: dict[int, dict[str, Any]] = {}
        for track_id, seed in enumerate(seeds):
            track_prompts[track_id] = seed["prompt"]
            track_seed_metadata[track_id] = {
                "semantic_query": seed["semantic_query"],
                "seed_semantic_score": seed["score"],
            }
            if "classification_rule" in seed:
                track_seed_metadata[track_id]["classification_rule"] = seed[
                    "classification_rule"
                ]
            seed["track_id"] = track_id
            predictor.handle_request(
                {
                    "type": "add_prompt",
                    "session_id": session_id,
                    "frame_index": prompt_frame,
                    "points": [seed["seed_point_xy_rel"]],
                    "point_labels": [1],
                    "obj_id": track_id,
                }
            )
        postprocess_audit["final_tracks"] = [
            {
                key: value
                for key, value in seed.items()
                if key
                in {
                    "track_id",
                    "prompt",
                    "semantic_query",
                    "classification_rule",
                    "score",
                    "bbox_xyxy_norm1000",
                    "bbox_yxyx_256",
                    "centroid_xy_rel",
                    "seed_point_xy_rel",
                }
            }
            for seed in seeds
        ]
        prompt_summaries["postprocessing"] = postprocess_audit

        per_frame: dict[int, list[dict[str, Any]]] = {}
        for response in predictor.handle_stream_request(
            {
                "type": "propagate_in_video",
                "session_id": session_id,
                "propagation_direction": "forward",
                "start_frame_index": prompt_frame,
                "output_prob_thresh": score_threshold,
            }
        ):
            frame_index = int(response["frame_index"])
            if frame_index < 0 or frame_index >= len(annotated_frames):
                continue
            per_frame[frame_index] = annotate_output(
                track_prompts=track_prompts,
                track_seed_metadata=track_seed_metadata,
                frame_index=frame_index,
                output=response["outputs"],
                frame=annotated_frames[frame_index],
                alpha=mask_alpha,
            )
    finally:
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "clear_cache_threshold": 95,
            }
        )
    return per_frame, prompt_summaries, track_prompts, time.monotonic() - started


def save_video(
    frames: list[np.ndarray], fps: float, output_path: Path, keep_mp4v: bool
) -> str:
    height, width = frames[0].shape[:2]
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=output_path.parent) as tmp_dir:
        mp4v_path = Path(tmp_dir) / "segmented_mp4v.mp4"
        writer = cv2.VideoWriter(
            str(mp4v_path),
            cv2.VideoWriter_fourcc(*"mp4v"),
            fps,
            (width, height),
        )
        if not writer.isOpened():
            raise RuntimeError("OpenCV could not create the output video.")
        for index, frame in enumerate(frames):
            cv2.putText(
                frame,
                f"frame {index}",
                (8, 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.48,
                (255, 255, 255),
                1,
                cv2.LINE_AA,
            )
            writer.write(frame)
        writer.release()

        if keep_mp4v:
            output_path.write_bytes(mp4v_path.read_bytes())
            return "mp4v"
        try:
            import imageio_ffmpeg

            ffmpeg = imageio_ffmpeg.get_ffmpeg_exe()
            subprocess.run(
                [
                    ffmpeg,
                    "-y",
                    "-loglevel",
                    "error",
                    "-i",
                    str(mp4v_path),
                    "-c:v",
                    "libx264",
                    "-crf",
                    "18",
                    "-pix_fmt",
                    "yuv420p",
                    "-movflags",
                    "+faststart",
                    str(output_path),
                ],
                check=True,
            )
            return "h264"
        except Exception as exc:
            output_path.write_bytes(mp4v_path.read_bytes())
            print(f"Warning: H.264 re-encode failed; kept mp4v output: {exc}")
            return "mp4v"


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SAM video smoke test.")
    if not args.video.is_file():
        raise FileNotFoundError(args.video)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    if not 0.0 <= args.mask_alpha <= 1.0:
        raise ValueError("--mask-alpha must be in [0, 1].")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    annotated_frames, fps = load_video(args.video)
    height, width = annotated_frames[0].shape[:2]
    if not 0 <= args.prompt_frame < len(annotated_frames):
        raise ValueError("--prompt-frame is outside the video.")

    print(
        f"Loading {args.model_version} on {torch.cuda.get_device_name(0)}; "
        f"video={len(annotated_frames)} frames, {width}x{height}, {fps:.3f} fps"
    )
    predictor = build_predictor(args)

    detections_by_frame: dict[int, list[dict[str, Any]]] = defaultdict(list)
    active_query_specs = (
        BUTTON_UNMASK_SWAP_QUERY_SPECS
        if args.button_unmask_swap_layout
        else [(prompt, prompt) for prompt in args.prompts]
    )
    print(f"Grounding prompt-frame instances for: {active_query_specs}")
    joint_outputs, prompt_summaries, track_prompts, elapsed = run_joint_tracking(
        predictor=predictor,
        video=args.video,
        prompts=args.prompts,
        prompt_frame=args.prompt_frame,
        score_threshold=args.score_threshold,
        annotated_frames=annotated_frames,
        mask_alpha=args.mask_alpha,
        button_unmask_swap_layout=args.button_unmask_swap_layout,
    )
    for frame_index, detections in joint_outputs.items():
        detections_by_frame[frame_index].extend(detections)
    nonempty = {
        frame_index: detections
        for frame_index, detections in joint_outputs.items()
        if detections
    }
    joint_summary = {
        "elapsed_seconds": round(elapsed, 3),
        "frames_with_detections": len(nonempty),
        "first_detection_frame": min(nonempty) if nonempty else None,
        "last_detection_frame": max(nonempty) if nonempty else None,
        "max_instances_in_one_frame": max(
            (len(detections) for detections in nonempty.values()), default=0
        ),
        "track_prompts": {str(key): value for key, value in track_prompts.items()},
    }
    print(json.dumps(joint_summary, ensure_ascii=False))

    jsonl_path = args.output_dir / "detections.jsonl"
    text_path = args.output_dir / "detections.txt"
    with jsonl_path.open("w", encoding="utf-8") as jsonl, text_path.open(
        "w", encoding="utf-8"
    ) as text:
        text.write(
            "Coordinates: bbox_xyxy_norm1000=[x1,y1,x2,y2]; "
            "bbox_yxyx_256=[y1,x1,y2,x2]\n"
        )
        for frame_index in range(len(annotated_frames)):
            record = {
                "frame_index": frame_index,
                "timestamp_seconds": round(frame_index / fps, 6),
                "detections": detections_by_frame.get(frame_index, []),
            }
            jsonl.write(json.dumps(record, ensure_ascii=False) + "\n")
            text.write(
                f"frame={frame_index:04d} "
                f"time={record['timestamp_seconds']:.3f}s "
                f"detections={len(record['detections'])}\n"
            )
            for detection in record["detections"]:
                text.write(
                    "  "
                    f"{detection['prompt']}:{detection['track_id']} "
                    f"tracking_score={detection['score']} "
                    f"seed_semantic_score={detection['seed_semantic_score']} "
                    f"seed_query={detection['seed_semantic_query']!r} "
                    f"xyxy1000={detection['bbox_xyxy_norm1000']} "
                    f"yxyx256={detection['bbox_yxyx_256']} "
                    f"area={detection['mask_area_pixels']}\n"
                )

    video_path = args.output_dir / "segmented.mp4"
    codec = save_video(annotated_frames, fps, video_path, args.keep_mp4v)
    summary = {
        "model_version": args.model_version,
        "checkpoint": str(args.checkpoint.resolve()),
        "input_video": str(args.video.resolve()),
        "prompts": args.prompts,
        "query_specs": [
            {"output_class": label, "semantic_query": query}
            for label, query in active_query_specs
        ],
        "prompt_frame": args.prompt_frame,
        "score_threshold": args.score_threshold,
        "button_unmask_swap_layout": args.button_unmask_swap_layout,
        "frame_count": len(annotated_frames),
        "fps": fps,
        "width": width,
        "height": height,
        "output_codec": codec,
        "seed_results": prompt_summaries,
        "joint_tracking": joint_summary,
        "outputs": {
            "segmented_video": str(video_path.resolve()),
            "jsonl": str(jsonl_path.resolve()),
            "text": str(text_path.resolve()),
        },
    }
    summary_path = args.output_dir / "summary.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
