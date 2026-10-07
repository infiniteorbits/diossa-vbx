#!/usr/bin/env python
"""Compare on-device predictions in pred/ with ground truth in gt/.

MobilePose writes keypoint coordinates on the heatmap (56x56). Those are
scaled onto the network input (224x224) before the Euclidean error is
measured. FCOS writes boxes already in input-image pixels. Each image
has one object, so the highest-scoring detection is compared with that
ground-truth box by IoU.

Each comparison also writes pred_annot/. Ground truth is green and the
prediction is red. Keypoint samples show the points. Detection samples
show the one ground-truth box and the highest-scoring box. ``--pair-lines``
draws a yellow line from each ground-truth keypoint to its prediction.

Usage:
    python -m diossa_model_exporter.compare-sample \
        --sample-dir output/images/ccn1-kr-224x224-test-sample \
        --task keypoints --pair-lines

    python -m diossa_model_exporter.compare-sample \
        --sample-dir output/images/ccn1-od-384x288-test-sample \
        --task boxes
"""

import argparse
import json
import pathlib
from typing import Any, Optional

import cv2
import numpy as np

PCK_THRESHOLDS_PX = (5.0, 10.0)
IOU_THRESHOLDS = (0.5, 0.75)


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare pred/ JSON tensors with gt/ keypoints or boxes.",
    )
    parser.add_argument(
        "--sample-dir", type=pathlib.Path, required=True,
        help="Sample directory containing pred/ and gt/",
    )
    parser.add_argument(
        "--task", choices=("keypoints", "boxes"), default=None,
        help="Comparison to run. Default: infer from the prediction tensor names",
    )
    parser.add_argument(
        "--input-width", type=int, default=224,
        help="Network input width. Keypoint coordinates are scaled onto this (default: 224)",
    )
    parser.add_argument(
        "--input-height", type=int, default=224,
        help="Network input height. Keypoint coordinates are scaled onto this (default: 224)",
    )
    parser.add_argument(
        "--heatmap-width", type=int, default=None,
        help="Heatmap width. Default: read from the heatmaps tensor, else 56",
    )
    parser.add_argument(
        "--heatmap-height", type=int, default=None,
        help="Heatmap height. Default: read from the heatmaps tensor, else 56",
    )
    parser.add_argument(
        "--score-thresh", type=float, default=0.0,
        help="Drop FCOS boxes at or below this score. Padding boxes are 0 (default: 0)",
    )
    parser.add_argument(
        "--output", type=pathlib.Path, default=None,
        help="Metrics JSON path (default: <sample-dir>/metrics.json)",
    )
    parser.add_argument(
        "--pair-lines", action="store_true",
        help="Draw a line between each ground-truth keypoint and its prediction",
    )
    return parser.parse_args()


def load_named_outputs(path: pathlib.Path) -> dict[str, dict[str, Any]]:
    """Load a run-model JSON file and index tensors by name."""
    with path.open() as handle:
        payload = json.load(handle)
    outputs = payload.get("outputs")
    if not isinstance(outputs, list):
        raise ValueError(f"{path} has no outputs list")
    return {item["name"]: item for item in outputs}


def as_array(output: dict[str, Any]) -> np.ndarray:
    """Reshape one JSON tensor into a float array."""
    return np.asarray(output["data"], dtype=np.float64).reshape(output["shape"])


def prediction_pairs(sample_dir: pathlib.Path) -> tuple[list[tuple[pathlib.Path, pathlib.Path]], list[str]]:
    """Pair each pred JSON with the ground-truth JSON of the same name."""
    pred_dir = sample_dir / "pred"
    gt_dir = sample_dir / "gt"
    if not pred_dir.is_dir():
        raise FileNotFoundError(f"No predictions in {pred_dir}")
    if not gt_dir.is_dir():
        raise FileNotFoundError(f"No ground truth in {gt_dir}")

    pairs = []
    missing = []
    for pred_path in sorted(pred_dir.glob("*.json")):
        gt_path = gt_dir / pred_path.name
        if not gt_path.is_file():
            missing.append(pred_path.name)
            continue
        pairs.append((pred_path, gt_path))
    if not pairs:
        raise FileNotFoundError(f"No pred/gt pairs under {sample_dir}")
    return pairs, missing


def infer_task(sample_dir: pathlib.Path) -> str:
    """Choose keypoints or boxes from the first prediction file."""
    pred_dir = sample_dir / "pred"
    pred_paths = sorted(pred_dir.glob("*.json"))
    if not pred_paths:
        raise FileNotFoundError(f"No predictions in {pred_dir}")
    names = set(load_named_outputs(pred_paths[0]))
    if "coords" in names:
        return "keypoints"
    if "boxes" in names and "scores" in names:
        return "boxes"
    raise ValueError(
        f"Cannot tell which comparison to run from {pred_paths[0].name}. "
        "Pass --task keypoints or --task boxes."
    )


def visible_keypoints(raw: list[Any]) -> tuple[np.ndarray, np.ndarray]:
    """Return xy keypoints and a visibility mask.

    A third value at or below 0 marks an invisible keypoint.
    """
    points = np.asarray(raw, dtype=np.float64)
    if points.ndim != 2 or points.shape[1] < 2:
        raise ValueError(f"Expected keypoints with shape (K, 2+), got {points.shape}")
    if points.shape[1] == 2:
        visible = np.ones(points.shape[0], dtype=bool)
    else:
        visible = points[:, 2] > 0
    return points[:, :2], visible


def heatmap_size(outputs: dict[str, dict[str, Any]], args: argparse.Namespace) -> tuple[int, int]:
    """Heatmap height and width, from the CLI or the heatmaps tensor."""
    height = args.heatmap_height
    width = args.heatmap_width
    heatmaps = outputs.get("heatmaps")
    if heatmaps is not None:
        shape = heatmaps["shape"]
        if height is None:
            height = int(shape[-2])
        if width is None:
            width = int(shape[-1])
    if height is None:
        height = 56
    if width is None:
        width = 56
    if height < 1 or width < 1:
        raise ValueError(f"Invalid heatmap size {width}x{height}")
    return height, width


def scale_coords(coords: np.ndarray, input_hw: tuple[int, int], heatmap_hw: tuple[int, int]) -> np.ndarray:
    """Map DSNT heatmap coordinates onto the network input image.

    ``normalized_coordinates=false`` leaves coordinates on the heatmap grid,
    ordered (x, y). Ground truth is stored in input pixels, so each axis is
    multiplied by input_size / heatmap_size (224/56 for the 224 model).
    """
    input_h, input_w = input_hw
    heatmap_h, heatmap_w = heatmap_hw
    scaled = np.array(coords, dtype=np.float64, copy=True)
    scaled[..., 0] *= input_w / heatmap_w
    scaled[..., 1] *= input_h / heatmap_h
    return scaled


GT_COLOR = (0, 255, 0)
PRED_COLOR = (0, 0, 255)
PAIR_COLOR = (0, 255, 255)


def draw_box(canvas: np.ndarray, box: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw one xyxy box."""
    x1, y1, x2, y2 = (int(round(float(value))) for value in box[:4])
    cv2.rectangle(canvas, (x1, y1), (x2, y2), color, 1)


def draw_pair_lines(
    canvas: np.ndarray,
    gt_keypoints: np.ndarray,
    pred_keypoints: np.ndarray,
    visible: Optional[np.ndarray] = None,
) -> None:
    """Draw a line from each ground-truth keypoint to the prediction with the same index."""
    count = min(len(gt_keypoints), len(pred_keypoints))
    for index in range(count):
        if visible is not None and index < len(visible) and not bool(visible[index]):
            continue
        start = (
            int(round(float(gt_keypoints[index, 0]))),
            int(round(float(gt_keypoints[index, 1]))),
        )
        end = (
            int(round(float(pred_keypoints[index, 0]))),
            int(round(float(pred_keypoints[index, 1]))),
        )
        cv2.line(canvas, start, end, PAIR_COLOR, 1, cv2.LINE_AA)


def draw_keypoints(canvas: np.ndarray, keypoints: np.ndarray, color: tuple[int, int, int]) -> None:
    """Draw numbered keypoints."""
    for index, point in enumerate(keypoints):
        x, y = int(round(float(point[0]))), int(round(float(point[1])))
        cv2.circle(canvas, (x, y), 1, color, -1)
        cv2.putText(
            canvas,
            str(index),
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            color,
            1,
            cv2.LINE_AA,
        )


def save_pred_annot(
    sample_dir: pathlib.Path,
    filename: str,
    gt_keypoints: Optional[np.ndarray],
    pred_keypoints: Optional[np.ndarray],
    gt_box: Optional[np.ndarray],
    pred_box: Optional[np.ndarray],
    pair_lines: bool = False,
    visible: Optional[np.ndarray] = None,
) -> None:
    """Draw ground truth in green and the prediction in red, then save pred_annot/."""
    stem = pathlib.Path(filename).stem
    raw_path = sample_dir / "raw" / f"{stem}.jpg"
    image = cv2.imread(str(raw_path), cv2.IMREAD_COLOR)
    if image is None:
        raise ValueError(f"Could not read {raw_path}")
    
    canvas = image.copy()
    if gt_box is not None:
        draw_box(canvas, gt_box, GT_COLOR)
    if pair_lines and gt_keypoints is not None and pred_keypoints is not None:
        draw_pair_lines(canvas, gt_keypoints, pred_keypoints, visible)
    if gt_keypoints is not None:
        draw_keypoints(canvas, gt_keypoints, GT_COLOR)
    if pred_box is not None:
        draw_box(canvas, pred_box, PRED_COLOR)
    if pred_keypoints is not None:
        draw_keypoints(canvas, pred_keypoints, PRED_COLOR)

    out_dir = sample_dir / "pred_annot"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{stem}.jpg"
    if not cv2.imwrite(str(out_path), canvas):
        raise ValueError(f"Failed to write {out_path}")


def compare_keypoints(args: argparse.Namespace) -> dict[str, Any]:
    """Euclidean error of scaled MobilePose coordinates against gt keypoints."""
    pairs, missing = prediction_pairs(args.sample_dir)
    per_image = []
    all_errors = []
    keypoint_errors: list[list[float]] = []
    scale = None

    for pred_path, gt_path in pairs:
        outputs = load_named_outputs(pred_path)
        if "coords" not in outputs:
            raise ValueError(f"{pred_path.name} has no coords output")
        coords = as_array(outputs["coords"])
        coords = np.squeeze(coords, axis=0) if coords.ndim == 3 and coords.shape[0] == 1 else coords
        if coords.ndim != 2 or coords.shape[1] != 2:
            raise ValueError(f"{pred_path.name} coords have shape {coords.shape}, expected (K, 2)")

        heatmap_hw = heatmap_size(outputs, args)
        input_hw = (args.input_height, args.input_width)
        image_scale = (input_hw[1] / heatmap_hw[1], input_hw[0] / heatmap_hw[0])
        if scale is None:
            scale = {"x": image_scale[0], "y": image_scale[1], "heatmap_hw": list(heatmap_hw), "input_hw": list(input_hw)}
        elif image_scale != (scale["x"], scale["y"]):
            raise ValueError(f"{pred_path.name} heatmap size differs from the first image")

        predicted = scale_coords(coords, input_hw, heatmap_hw)
        with gt_path.open() as handle:
            ground_truth = json.load(handle)
        target, visible = visible_keypoints(ground_truth["keypoints"])
        save_pred_annot(
            args.sample_dir, pred_path.name, target, predicted, None, None,
            pair_lines=args.pair_lines, visible=visible,
        )
        if predicted.shape[0] != target.shape[0]:
            raise ValueError(
                f"{pred_path.name} has {predicted.shape[0]} keypoints, "
                f"ground truth has {target.shape[0]}"
            )

        if keypoint_errors and len(keypoint_errors) != predicted.shape[0]:
            raise ValueError(f"{pred_path.name} keypoint count changed mid-sample")
        if not keypoint_errors:
            keypoint_errors = [[] for _ in range(predicted.shape[0])]
        errors = np.linalg.norm(predicted - target, axis=1)
        for index, error in enumerate(errors):
            if visible[index]:
                keypoint_errors[index].append(float(error))
                all_errors.append(float(error))
        visible_errors = errors[visible]
        per_image.append({
            "file": pred_path.name,
            "mean_px": float(visible_errors.mean()) if visible_errors.size else None,
            "visible": int(visible.sum()),
        })

    if not all_errors:
        raise ValueError(f"No visible keypoints under {args.sample_dir}")
    errors_array = np.asarray(all_errors, dtype=np.float64)
    per_keypoint_mean = [
        float(np.mean(values)) if values else None
        for values in keypoint_errors
    ]
    summary = {
        "task": "keypoints",
        "images": len(per_image),
        "missing_ground_truth": missing,
        "scale": scale,
        "keypoints": int(errors_array.size),
        "mean_euclidean_px": float(errors_array.mean()),
        "median_euclidean_px": float(np.median(errors_array)),
        "pck": {
            str(int(threshold) if threshold.is_integer() else threshold): float(np.mean(errors_array <= threshold))
            for threshold in PCK_THRESHOLDS_PX
        },
        "per_keypoint_mean_px": per_keypoint_mean,
        "per_image": per_image,
    }
    return summary


def box_iou(pred: np.ndarray, ground_truth: np.ndarray) -> float:
    """IoU of two xyxy boxes."""
    left_top = np.maximum(pred[:2], ground_truth[:2])
    right_bottom = np.minimum(pred[2:], ground_truth[2:])
    width_height = np.clip(right_bottom - left_top, 0, None)
    intersection = float(width_height[0] * width_height[1])
    pred_area = float(max(pred[2] - pred[0], 0.0) * max(pred[3] - pred[1], 0.0))
    gt_area = float(max(ground_truth[2] - ground_truth[0], 0.0) * max(ground_truth[3] - ground_truth[1], 0.0))
    union = pred_area + gt_area - intersection
    if union <= 0.0:
        return 0.0
    return intersection / union


def kept_detections(outputs: dict[str, dict[str, Any]], score_thresh: float) -> tuple[np.ndarray, np.ndarray]:
    """Drop zero-padded FCOS boxes and anything at or below the score threshold."""
    if "boxes" not in outputs or "scores" not in outputs:
        raise ValueError("Prediction is missing boxes or scores")
    boxes = as_array(outputs["boxes"]).reshape(-1, 4)
    scores = as_array(outputs["scores"]).reshape(-1)
    if boxes.shape[0] != scores.shape[0]:
        raise ValueError(f"boxes ({boxes.shape[0]}) and scores ({scores.shape[0]}) differ in length")
    positive_area = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
    keep = (scores > score_thresh) & positive_area
    return boxes[keep], scores[keep]


def compare_boxes(args: argparse.Namespace) -> dict[str, Any]:
    """IoU of the highest-scoring FCOS box against the one ground-truth box."""
    pairs, missing = prediction_pairs(args.sample_dir)
    per_image = []
    ious = []

    for pred_path, gt_path in pairs:
        outputs = load_named_outputs(pred_path)
        boxes, scores = kept_detections(outputs, args.score_thresh)
        with gt_path.open() as handle:
            ground_truth = json.load(handle)
        gt_boxes = np.asarray(ground_truth["bboxes"], dtype=np.float64).reshape(-1, 4)
        if gt_boxes.shape[0] != 1:
            raise ValueError(f"{gt_path.name} has {gt_boxes.shape[0]} boxes; expected 1")

        if boxes.shape[0] == 0:
            iou = 0.0
            score = None
            box = None
        else:
            top_index = int(np.argmax(scores))
            box = boxes[top_index]
            iou = box_iou(box, gt_boxes[0])
            score = float(scores[top_index])
        save_pred_annot(args.sample_dir, pred_path.name, None, None, gt_boxes[0], box)
        ious.append(iou)
        per_image.append({
            "file": pred_path.name,
            "iou": iou,
            "score": score,
        })

    measured = np.asarray(ious, dtype=np.float64)
    return {
        "task": "boxes",
        "images": len(per_image),
        "missing_ground_truth": missing,
        "score_thresh": args.score_thresh,
        "mean_iou": float(measured.mean()),
        "median_iou": float(np.median(measured)),
        "recall": {
            f"{threshold:.2f}": float(np.mean(measured >= threshold))
            for threshold in IOU_THRESHOLDS
        },
        "per_image": per_image,
    }


def print_summary(summary: dict[str, Any]) -> None:
    """Print the aggregate comparison."""
    print(f"Images: {summary['images']}")
    missing = summary["missing_ground_truth"]
    if missing:
        print(f"Predictions without ground truth: {len(missing)}")
    if summary["task"] == "keypoints":
        scale = summary["scale"]
        print(
            f"Scale: heatmap {scale['heatmap_hw'][1]}x{scale['heatmap_hw'][0]} "
            f"-> input {scale['input_hw'][1]}x{scale['input_hw'][0]} "
            f"(x{scale['x']:g}, y{scale['y']:g})"
        )
        print(f"Mean Euclidean error: {summary['mean_euclidean_px']:.3f} px")
        print(f"Median Euclidean error: {summary['median_euclidean_px']:.3f} px")
        for threshold, value in summary["pck"].items():
            print(f"PCK@{threshold}px: {value:.3f}")
        print("Per-keypoint mean error (px):")
        for index, value in enumerate(summary["per_keypoint_mean_px"]):
            rendered = "n/a" if value is None else f"{value:.3f}"
            print(f"  {index:2d}  {rendered}")
        return

    print(f"Mean IoU: {summary['mean_iou']:.3f}")
    print(f"Median IoU: {summary['median_iou']:.3f}")
    for threshold, value in summary["recall"].items():
        print(f"Recall @ IoU {threshold}: {value:.3f}")


def main() -> None:
    """Compare one sample directory and write metrics.json."""
    args = parse_args()
    if args.task is None:
        args.task = infer_task(args.sample_dir)
    if args.task == "keypoints":
        summary = compare_keypoints(args)
    else:
        summary = compare_boxes(args)

    output_path = args.output or (args.sample_dir / "metrics.json")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w") as handle:
        json.dump(summary, handle)
        handle.write("\n")

    print_summary(summary)
    print(f"Wrote {args.sample_dir / 'pred_annot'}")
    print(f"Wrote {output_path}")


if __name__ == "__main__":
    main()
