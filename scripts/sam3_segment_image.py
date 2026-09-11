#!/usr/bin/env python3
"""Run SAM 3.1 semantic segmentation on one image and merge class results."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import torch

from sam3_segment_video import (
    PROMPT_COLORS_BGR,
    build_predictor,
    mask_record,
    to_numpy,
)

QUERY_SPECS = [
    ("cube", "colored cube"),
    ("button", "white square button"),
    ("robot arm", "robot"),
    ("robot arm", "robot arm"),
    ("robot arm", "robot gripper"),
    ("robot arm", "gripper"),
    ("robot arm", "robotic gripper"),
    ("robot arm", "robot end effector"),
]
CLASS_LABELS = ("cube", "container", "button", "robot arm")
CLASS_PRIORITY = {"robot arm": 0, "cube": 1, "container": 2, "button": 3}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--model-version", choices=("sam3.1", "sam3"), default="sam3.1")
    parser.add_argument("--score-threshold", type=float, default=0.4)
    parser.add_argument("--mask-alpha", type=float, default=0.42)
    return parser.parse_args()


def semantic_detections(
    *,
    predictor: Any,
    session_id: str,
    label: str,
    query: str,
    score_threshold: float,
    next_track_id: int,
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    response = predictor.handle_request(
        {
            "type": "add_prompt",
            "session_id": session_id,
            "frame_index": 0,
            "text": query,
            "output_prob_thresh": score_threshold,
        }
    )
    output = response["outputs"]
    masks = to_numpy(output.get("out_binary_masks", np.empty((0,))))
    probs_raw = output.get("out_probs")
    probs = None if probs_raw is None else to_numpy(probs_raw)
    if masks.ndim == 4 and masks.shape[1] == 1:
        masks = masks[:, 0]

    records: list[dict[str, Any]] = []
    bool_masks: list[np.ndarray] = []
    for index, raw_mask in enumerate(masks):
        mask = np.asarray(raw_mask).astype(bool)
        score = None
        if probs is not None and index < len(probs):
            score = float(np.asarray(probs[index]).reshape(-1)[0])
        record = mask_record(
            prompt=label,
            frame_index=0,
            track_id=next_track_id + len(records),
            score=score,
            mask=mask,
        )
        if record is not None:
            record["semantic_query"] = query
            records.append(record)
            bool_masks.append(mask)
    return records, bool_masks


def draw_detection(
    frame: np.ndarray,
    record: dict[str, Any],
    mask: np.ndarray,
    alpha: float,
) -> None:
    label = record["prompt"]
    color = np.asarray(PROMPT_COLORS_BGR[label], dtype=np.uint8)
    frame[mask] = ((1.0 - alpha) * frame[mask] + alpha * color).astype(np.uint8)
    x1, y1, x2, y2 = record["bbox_xyxy_pixels"]
    text = f"{label}:{record['track_id']}"
    if record["score"] is not None:
        text += f" {record['score']:.2f}"
    cv2.rectangle(frame, (x1, y1), (x2, y2), color.tolist(), 1)
    cv2.putText(
        frame,
        text,
        (x1, max(12, y1 - 4)),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.38,
        color.tolist(),
        1,
        cv2.LINE_AA,
    )


def resolve_cross_class_duplicates(
    records: list[dict[str, Any]], masks: list[np.ndarray]
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    ranked_indices = sorted(
        range(len(records)),
        key=lambda index: (
            CLASS_PRIORITY[records[index]["prompt"]],
            records[index]["score"] if records[index]["score"] is not None else -1.0,
        ),
        reverse=True,
    )
    retained_indices: list[int] = []
    for index in ranked_indices:
        mask = masks[index]
        area = int(mask.sum())
        if area == 0:
            continue
        is_duplicate = False
        for retained_index in retained_indices:
            retained_mask = masks[retained_index]
            intersection = int(np.logical_and(mask, retained_mask).sum())
            smaller_area = min(area, int(retained_mask.sum()))
            if smaller_area > 0 and intersection / smaller_area >= 0.8:
                is_duplicate = True
                break
        if not is_duplicate:
            retained_indices.append(index)

    retained_indices.sort()
    filtered_records = [records[index] for index in retained_indices]
    filtered_masks = [masks[index] for index in retained_indices]
    for track_id, record in enumerate(filtered_records):
        record["track_id"] = track_id
    return filtered_records, filtered_masks


def main() -> None:
    args = parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for this SAM image smoke test.")
    if not args.image.is_file():
        raise FileNotFoundError(args.image)
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)

    frame = cv2.imread(str(args.image), cv2.IMREAD_COLOR)
    if frame is None:
        raise RuntimeError(f"Could not read image: {args.image}")
    args.output_dir.mkdir(parents=True, exist_ok=True)

    predictor = build_predictor(args)
    response = predictor.handle_request(
        {"type": "start_session", "resource_path": str(args.image)}
    )
    session_id = response["session_id"]
    records: list[dict[str, Any]] = []
    masks: list[np.ndarray] = []
    try:
        for label, query in QUERY_SPECS:
            class_records, class_masks = semantic_detections(
                predictor=predictor,
                session_id=session_id,
                label=label,
                query=query,
                score_threshold=args.score_threshold,
                next_track_id=len(records),
            )
            records.extend(class_records)
            masks.extend(class_masks)
    finally:
        predictor.handle_request(
            {
                "type": "close_session",
                "session_id": session_id,
                "clear_cache_threshold": 95,
            }
        )

    raw_counts_by_query = {
        query: sum(record["semantic_query"] == query for record in records)
        for _, query in QUERY_SPECS
    }
    for record in records:
        if record["prompt"] == "button" and record["centroid_xy_pixels"][1] >= 390:
            record["prompt"] = "container"
            record["classification_rule"] = "lower white square movable object"
    raw_counts = {
        label: sum(record["prompt"] == label for record in records)
        for label in CLASS_LABELS
    }
    records, masks = resolve_cross_class_duplicates(records, masks)
    annotated = frame.copy()
    for record, mask in zip(records, masks):
        draw_detection(annotated, record, mask, args.mask_alpha)

    image_path = args.output_dir / "segmented.png"
    detections_path = args.output_dir / "detections.json"
    summary_path = args.output_dir / "summary.json"
    if not cv2.imwrite(str(image_path), annotated):
        raise RuntimeError(f"Could not write image: {image_path}")
    detections_path.write_text(
        json.dumps({"frame_index": 0, "detections": records}, indent=2) + "\n",
        encoding="utf-8",
    )
    counts = {
        label: sum(record["prompt"] == label for record in records)
        for label in CLASS_LABELS
    }
    summary = {
        "model_version": args.model_version,
        "checkpoint": str(args.checkpoint.resolve()),
        "input_image": str(args.image.resolve()),
        "query_specs": [
            {"output_class": label, "semantic_query": query}
            for label, query in QUERY_SPECS
        ],
        "white_object_classification": {
            "button": "centroid y < 390",
            "container": "centroid y >= 390",
        },
        "score_threshold": args.score_threshold,
        "width": int(frame.shape[1]),
        "height": int(frame.shape[0]),
        "raw_instance_counts_by_query": raw_counts_by_query,
        "raw_instance_counts": raw_counts,
        "instance_counts": counts,
        "total_instances": len(records),
        "outputs": {
            "segmented_image": str(image_path.resolve()),
            "detections": str(detections_path.resolve()),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
