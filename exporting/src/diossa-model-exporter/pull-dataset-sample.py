#!/usr/bin/env python
# coding: utf-8

# In[129]:


from typing import Any
import requests as req
import json

import pandas as pd
import cv2
import numpy as np


HTTP_TIMEOUT = 3


# In[130]:


# Datasets Viewer configuration and connection test

DATASET_VIEWER_URL = "http://datasets.int.lmo.space"


def fetch_data_from_dv(
    endpoint: str,
    params: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Fetch data (using GET) from the Dataset Viewer.

    Args:
        endpoint (str): The endpoint to fetch data from.
        params (dict[str, Any], optional): The parameters to pass to the endpoint.
    """
    response = req.get(
        f"{DATASET_VIEWER_URL}/{endpoint}",
        params=params,
        timeout=HTTP_TIMEOUT,
    )
    if response.status_code != 200:
        raise ValueError(
            f"Failed to fetch data from {endpoint}. Response was: {response.text}"
        )
    try:
        response_parsed: dict[str, Any] = json.loads(response.content)
    except json.JSONDecodeError as err:
        raise ValueError(
            f"Failed to parse response from {endpoint}. "
            "The response was: {response.text}",
        ) from err
    return response_parsed


assert len(fetch_data_from_dv("api/v1/datasets/versions")) > 0, (
    "No versions found. Make sure you are on the VPN and that your DNS can resolve "
    f"{DATASET_VIEWER_URL}"
)


# In[131]:


pipelines_raw = fetch_data_from_dv("api/v1/pipelines")

pipelines_df = pd.DataFrame(pipelines_raw)
pipelines_df


# In[132]:


pipeline_name = "[Keypoint Regression] AutoCrop -> Resize"
pipeline_id = pipelines_df.query("name == @pipeline_name").id.iloc[0]

f"Pipeline ID: {pipeline_id} for {pipeline_name}"


# In[133]:


all_dv_versions = fetch_data_from_dv("api/v1/datasets/versions")
all_dv_versions_df = pd.DataFrame(all_dv_versions)
all_dv_versions_df.head()

selected_branch = "main"

dv_dataset_version = all_dv_versions_df.query("reference_name == @selected_branch")
dv_dataset_commit_sha = dv_dataset_version.commit_sha.iloc[0]
dv_dataset_is_tag = dv_dataset_version.is_tag.iloc[0]

print(f"Selected branch: {selected_branch}")
print(f"Dataset version commit SHA: {dv_dataset_commit_sha}")
print(f"Dataset version is tag: {dv_dataset_is_tag}")


# In[134]:


dataset_name = "Inmarsat-5_DIOSSA-CCN_Pangu_v4"
dataset_path = "DIOSSA_CCN1"
split_name = "test"


# In[139]:


from tqdm import tqdm

sample_index = 0
pages = []
continue_loop = True
while continue_loop:
    page = fetch_data_from_dv(
        "api/v1/samples/page",
        params={
            "split_name": split_name,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "version_name": selected_branch,
            "version_commit": dv_dataset_commit_sha,
            "version_is_tag": dv_dataset_is_tag,
            "limit": 500,
            "after_sample_index": sample_index,
        },
    )
    df = pd.DataFrame(page)
    sample_index = df.sample_index.max()
    if len(df) < 500:
        continue_loop = False
    pages.append(df)

pages = pd.concat(pages)


# In[140]:


test_indices_sample = pages.sample_index.sample(100, random_state=42)
test_indices_sample.values


# In[152]:


expected_amount_of_gt_keypoints = 41

selected_keypoint_indices = [
    0,
    1,
    2,
    3,
    4,
    5,
    6,
    7,
    8,
    21,
    26,
    27,
    31,
    32,
    33,
    34,
    37,
    38,
    39,
    40,
]


# In[155]:


import pathlib, base64

output_file_path = pathlib.Path("ccn1-inmarsat-5-test-sample")
output_file_path.mkdir(parents=True, exist_ok=True)


for sample_index in test_indices_sample.values:
    fetched_details = fetch_data_from_dv(
        "api/v1/samples/detail",
        params={
            "split_name": split_name,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "version_name": selected_branch,
            "version_commit": dv_dataset_commit_sha,
            "version_is_tag": dv_dataset_is_tag,
            "sample_index": sample_index,
            "pipeline_id": pipeline_id,
        },
    )

    # Get the ground truth json from the Dataset Viewer
    filename = fetched_details["json_data"]["filename"]

    image_package = fetch_data_from_dv(
        "api/v1/samples/image",
        params={
            "split_name": split_name,
            "dataset_name": dataset_name,
            "dataset_path": dataset_path,
            "version_name": selected_branch,
            "version_commit": dv_dataset_commit_sha,
            "version_is_tag": dv_dataset_is_tag,
            "sample_index": sample_index,
            "pipeline_id": pipeline_id,
        },
    )
    image_b64 = image_package.get("content", None)
    if image_b64 is None:
        raise ValueError("Failed to take and image in SRS and download it")

    image_keypoints = image_package.get("targets", {}).get("keypoints", None)
    if len(image_keypoints) != expected_amount_of_gt_keypoints:
        print(
            f"Skipping sample where expected {expected_amount_of_gt_keypoints} keypoints "
            f"but got {len(image_keypoints)}. Sample index: {sample_index}. "
            "Some keypoint was likely outside FoV and dropped and we cannot guarantee "
            "that the remaining keypoints are correct"
        )
        continue

    buff = np.frombuffer(base64.b64decode(image_b64), np.uint8)
    decoded_array = cv2.imdecode(buff, cv2.IMREAD_COLOR)
    saved_image_path = (output_file_path / filename).with_suffix(
        ".jpg",
    )
    cv2.imwrite(saved_image_path, decoded_array)

    gt_keypoints = np.array(image_keypoints)[selected_keypoint_indices].tolist()

    with (output_file_path / filename).with_suffix(".gt.json").open("w") as f:
        json.dump({"keypoints": gt_keypoints}, f)


# In[ ]:




