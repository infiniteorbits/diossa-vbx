#!/usr/bin/env python
"""Download a random sample of images and ground truth from Dataset Viewer.

Each sample is written under --output-dir as:
    raw/<filename>.jpg    pipeline image
    gt/<filename>.json    selected keypoints and bounding boxes
    annot/<filename>.jpg  raw image with those boxes and keypoints drawn

Samples whose keypoint count does not match --expected-keypoints are skipped.

Usage:
    python pull-dataset-sample.py \
        --output-dir data/images/ccn1-inmarsat-5-test-sample \
        --dataset-name Inmarsat-5_DIOSSA-CCN_Pangu_v4 \
        --dataset-path DIOSSA_CCN1 \
        [--split-name test] \
        [--branch main] \
        [--pipeline-name "[Keypoint Regression] AutoCrop -> Resize to 224x224"] \
        [--sample-count 100] \
        [--seed 42] \
        [--expected-keypoints 41] \
        [--keypoint-indices 0 1 2 3 4 5 6 7 8 21 26 27 31 32 33 34 37 38 39 40]
"""

import argparse
import base64
import json
import pathlib
from typing import Any

import cv2
import numpy as np
import pandas as pd
import requests as req
from tqdm import tqdm

HTTP_TIMEOUT = 3
PAGE_SIZE = 500
DEFAULT_DATASET_VIEWER_URL = "http://datasets.int.lmo.space"
DEFAULT_PIPELINE_NAME = "[Keypoint Regression] AutoCrop -> Resize to 224x224"
DEFAULT_KEYPOINT_INDICES = [
    0, 1, 2, 3, 4, 5, 6, 7, 8,
    21, 26, 27, 31, 32, 33, 34, 37, 38, 39, 40,
]


def parse_args() -> argparse.Namespace:
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Download a random sample of images and ground truth "
        "from Dataset Viewer.",
    )
    parser.add_argument(
        "--output-dir", type=pathlib.Path, required=True,
        help="Directory where raw/, gt/, and annot/ are created",
    )
    parser.add_argument(
        "--dataset-name", type=str, required=True,
        help="Dataset Viewer dataset name",
    )
    parser.add_argument(
        "--dataset-path", type=str, required=True,
        help="Dataset path inside the Dataset Viewer dataset",
    )
    parser.add_argument(
        "--split-name", type=str, default="test",
        help="Dataset split to sample from (default: test)",
    )
    parser.add_argument(
        "--branch", type=str, default="main",
        help="Dataset version reference name, branch or tag (default: main)",
    )
    parser.add_argument(
        "--pipeline-name", type=str, default=DEFAULT_PIPELINE_NAME,
        help=f"Preprocessing pipeline name (default: {DEFAULT_PIPELINE_NAME})",
    )
    parser.add_argument(
        "--sample-count", type=int, default=100,
        help="Number of samples to draw (default: 100)",
    )
    parser.add_argument(
        "--seed", type=int, default=42,
        help="Random seed used when drawing samples (default: 42)",
    )
    parser.add_argument(
        "--expected-keypoints", type=int, default=41,
        help="Skip a sample unless it has exactly this many ground-truth keypoints "
        "(default: 41)",
    )
    parser.add_argument(
        "--keypoint-indices", type=int, nargs="+", default=DEFAULT_KEYPOINT_INDICES,
        help="Keypoint indices to keep, in order (default: the CCN1 subset)",
    )
    parser.add_argument(
        "--dataset-viewer-url", type=str, default=DEFAULT_DATASET_VIEWER_URL,
        help=f"Dataset Viewer base URL (default: {DEFAULT_DATASET_VIEWER_URL})",
    )
    parser.add_argument(
        "--timeout", type=float, default=HTTP_TIMEOUT,
        help=f"HTTP timeout in seconds (default: {HTTP_TIMEOUT})",
    )
    parser.add_argument(
        "--page-size", type=int, default=PAGE_SIZE,
        help=f"Sample-page size when listing the split (default: {PAGE_SIZE})",
    )
    return parser.parse_args()


def fetch_data_from_dv(
    base_url: str,
    endpoint: str,
    params: dict[str, Any] | None = None,
    timeout: float = HTTP_TIMEOUT,
) -> Any:
    """Fetch JSON from a Dataset Viewer GET endpoint."""
    response = req.get(
        f"{base_url.rstrip('/')}/{endpoint}",
        params=params,
        timeout=timeout,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch data from {endpoint}. Response was: {response.text}"
        )
    try:
        return json.loads(response.content)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Failed to parse response from {endpoint}. "
            f"The response was: {response.text}",
        ) from err


def sample_query_params(
    args: argparse.Namespace,
    commit_sha: str,
    is_tag: bool,
    sample_index: int | None = None,
    pipeline_id: Any = None,
) -> dict[str, Any]:
    """Build the query parameters shared by the sample endpoints."""
    params: dict[str, Any] = {
        "split_name": args.split_name,
        "dataset_name": args.dataset_name,
        "dataset_path": args.dataset_path,
        "version_name": args.branch,
        "version_commit": commit_sha,
        "version_is_tag": is_tag,
    }
    if sample_index is not None:
        params["sample_index"] = int(sample_index)
    if pipeline_id is not None:
        params["pipeline_id"] = pipeline_id
    return params


def resolve_pipeline_id(pipelines: pd.DataFrame, pipeline_name: str) -> Any:
    """Return the id of the named preprocessing pipeline."""
    matches = pipelines.query("name == @pipeline_name")
    if matches.empty:
        known = ", ".join(sorted(pipelines["name"].astype(str)))
        raise ValueError(
            f"No pipeline named {pipeline_name!r}. Known pipelines: {known}"
        )
    return matches.id.iloc[0]


def resolve_dataset_version(
    versions: pd.DataFrame,
    branch: str,
) -> tuple[str, bool]:
    """Return the commit SHA and tag flag for a dataset version reference."""
    matches = versions.query("reference_name == @branch")
    if matches.empty:
        known = ", ".join(sorted(versions["reference_name"].astype(str)))
        raise ValueError(
            f"No dataset version named {branch!r}. Known versions: {known}"
        )
    version = matches.iloc[0]
    return str(version.commit_sha), bool(version.is_tag)


def list_sample_indices(args: argparse.Namespace, commit_sha: str, is_tag: bool) -> pd.Series:
    """Page through the split and return every sample index."""
    pages: list[pd.DataFrame] = []
    after_sample_index = 0
    while True:
        page = fetch_data_from_dv(
            args.dataset_viewer_url,
            "api/v1/samples/page",
            params={
                **sample_query_params(args, commit_sha, is_tag),
                "limit": args.page_size,
                "after_sample_index": after_sample_index,
            },
            timeout=args.timeout,
        )
        df = pd.DataFrame(page)
        if df.empty:
            break
        pages.append(df)
        after_sample_index = df.sample_index.max()
        if len(df) < args.page_size:
            break
    if not pages:
        raise ValueError(
            f"No samples found for dataset {args.dataset_name!r} "
            f"path {args.dataset_path!r} split {args.split_name!r} "
            f"on {args.branch!r}."
        )
    return pd.concat(pages, ignore_index=True).sample_index


def _as_xyxy(box: Any) -> list[float] | None:
    """Return one box as [x1, y1, x2, y2]."""
    if isinstance(box, dict):
        corner_keys = (
            ("x1", "y1", "x2", "y2"),
            ("xmin", "ymin", "xmax", "ymax"),
            ("left", "top", "right", "bottom"),
        )
        for keys in corner_keys:
            if all(key in box for key in keys):
                return [float(box[key]) for key in keys]
        if all(key in box for key in ("x", "y", "w", "h")):
            x, y, w, h = (float(box[key]) for key in ("x", "y", "w", "h"))
            return [x, y, x + w, y + h]
        return None
    values = [float(value) for value in box]
    if len(values) < 4:
        return None
    return values[:4]


def normalize_bboxes(raw: Any) -> list[list[float]]:
    """Normalize a single box or a list of boxes to xyxy lists."""
    if raw is None:
        return []
    if isinstance(raw, dict):
        box = _as_xyxy(raw)
        return [box] if box is not None else []
    if not isinstance(raw, (list, tuple)) or len(raw) == 0:
        return []
    if isinstance(raw[0], (int, float)):
        box = _as_xyxy(raw)
        return [box] if box is not None else []
    return [box for box in (_as_xyxy(item) for item in raw) if box is not None]


def extract_bboxes(targets: dict[str, Any], json_data: dict[str, Any] | None) -> list[list[float]]:
    """Read bounding boxes from the pipeline targets, then from the sample record."""
    for source in (targets, json_data or {}):
        for key in ("bboxes", "boxes", "bbox", "bounding_boxes"):
            boxes = normalize_bboxes(source.get(key))
            if boxes:
                return boxes
    return []


def annotate_image(
    image: np.ndarray,
    keypoints: list[list[float]],
    bboxes: list[list[float]],
) -> np.ndarray:
    """Draw bounding boxes and keypoints on a copy of the image."""
    canvas = image.copy()
    for box in bboxes:
        x1, y1, x2, y2 = (int(round(value)) for value in box[:4])
        cv2.rectangle(canvas, (x1, y1), (x2, y2), (0, 255, 0), 1)
    for index, point in enumerate(keypoints):
        if len(point) > 2 and float(point[2]) <= 0:
            continue
        x, y = int(round(point[0])), int(round(point[1]))
        cv2.circle(canvas, (x, y), 1, (0, 255, 255), -1)
        cv2.putText(
            canvas,
            str(index),
            (x + 2, y - 2),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.3,
            (0, 255, 255),
            1,
            cv2.LINE_AA,
        )
    return canvas


def save_sample(
    args: argparse.Namespace,
    sample_index: int,
    commit_sha: str,
    is_tag: bool,
    pipeline_id: Any,
) -> bool:
    """Save one sample. Return False when the sample is skipped."""
    details = fetch_data_from_dv(
        args.dataset_viewer_url,
        "api/v1/samples/detail",
        params=sample_query_params(
            args, commit_sha, is_tag,
            sample_index=sample_index,
            pipeline_id=pipeline_id,
        ),
        timeout=args.timeout,
    )
    filename = details["json_data"]["filename"]

    image_package = fetch_data_from_dv(
        args.dataset_viewer_url,
        "api/v1/samples/image",
        params=sample_query_params(
            args, commit_sha, is_tag,
            sample_index=sample_index,
            pipeline_id=pipeline_id,
        ),
        timeout=args.timeout,
    )
    image_b64 = image_package.get("content", None)
    if image_b64 is None:
        raise ValueError(
            f"Sample {sample_index} did not include an image "
            f"(filename {filename!r})."
        )

    image_keypoints = image_package.get("targets", {}).get("keypoints", None)
    if image_keypoints is None or len(image_keypoints) != args.expected_keypoints:
        got = 0 if image_keypoints is None else len(image_keypoints)
        tqdm.write(
            f"Skipping sample where expected {args.expected_keypoints} keypoints "
            f"but got {got}. Sample index: {sample_index}. "
            "Some keypoint was likely outside FoV and dropped and we cannot guarantee "
            "that the remaining keypoints are correct"
        )
        return False

    decoded = cv2.imdecode(
        np.frombuffer(base64.b64decode(image_b64), np.uint8),
        cv2.IMREAD_COLOR,
    )
    if decoded is None:
        raise ValueError(f"Failed to decode image for sample {sample_index} ({filename}).")

    targets = image_package.get("targets") or {}
    bboxes = extract_bboxes(targets, details.get("json_data"))
    if not bboxes:
        tqdm.write(
            f"Sample {sample_index} has no bounding boxes; "
            "the annotated image will only show keypoints."
        )

    gt_keypoints = np.array(image_keypoints)[args.keypoint_indices].tolist()
    image_name = pathlib.Path(filename).with_suffix(".jpg").name
    gt_name = pathlib.Path(filename).with_suffix(".json").name
    raw_path = args.output_dir / "raw" / image_name
    gt_path = args.output_dir / "gt" / gt_name
    annot_path = args.output_dir / "annot" / image_name

    if not cv2.imwrite(str(raw_path), decoded):
        raise ValueError(f"Failed to write {raw_path}")
    with gt_path.open("w") as gt_file:
        json.dump({"keypoints": gt_keypoints, "bboxes": bboxes}, gt_file)
    annotated = annotate_image(decoded, gt_keypoints, bboxes)
    if not cv2.imwrite(str(annot_path), annotated):
        raise ValueError(f"Failed to write {annot_path}")
    return True


def main() -> None:
    """Download the requested sample from Dataset Viewer."""
    args = parse_args()
    if args.sample_count < 1:
        raise ValueError("--sample-count must be at least 1")
    if args.page_size < 1:
        raise ValueError("--page-size must be at least 1")

    versions = fetch_data_from_dv(
        args.dataset_viewer_url,
        "api/v1/datasets/versions",
        timeout=args.timeout,
    )
    if len(versions) == 0:
        raise ValueError(
            "No versions found. Make sure you are on the VPN and that your DNS can "
            f"resolve {args.dataset_viewer_url}"
        )

    pipelines = pd.DataFrame(
        fetch_data_from_dv(
            args.dataset_viewer_url,
            "api/v1/pipelines",
            timeout=args.timeout,
        )
    )
    pipeline_id = resolve_pipeline_id(pipelines, args.pipeline_name)
    commit_sha, is_tag = resolve_dataset_version(pd.DataFrame(versions), args.branch)
    print(f"Pipeline ID: {pipeline_id} for {args.pipeline_name}")
    print(f"Selected branch: {args.branch}")
    print(f"Dataset version commit SHA: {commit_sha}")
    print(f"Dataset version is tag: {is_tag}")

    sample_indices = list_sample_indices(args, commit_sha, is_tag)
    sample_count = min(args.sample_count, len(sample_indices))
    if sample_count < args.sample_count:
        print(
            f"Only {len(sample_indices)} samples available; "
            f"drawing {sample_count} instead of {args.sample_count}."
        )
    chosen = sample_indices.sample(sample_count, random_state=args.seed)

    for subdir in ("raw", "gt", "annot"):
        (args.output_dir / subdir).mkdir(parents=True, exist_ok=True)
    saved = 0
    for sample_index in tqdm(chosen.values, desc="Downloading samples"):
        if save_sample(args, int(sample_index), commit_sha, is_tag, pipeline_id):
            saved += 1
    print(f"Saved {saved} samples to {args.output_dir} ({sample_count - saved} skipped).")


if __name__ == "__main__":
    main()
