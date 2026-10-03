from pathlib import Path
import urllib.request

import pytest

import torch
import torchvision

from nn_models.pytorch.keypoints_regression import mobilepose
from nn_models.pytorch.object_detection import faster_rcnn, fcos


@pytest.fixture(scope="module")
def cache_path():
    dir = Path(".cache")
    if not dir.exists():
        dir.mkdir()
    if dir.is_dir():
        return dir
    else:
        pytest.fail("Error accessing cache directory!")


def test_fasterrcnn_ots_loader(device, cache_path):
    model_url = torchvision.models.detection.FasterRCNN_ResNet50_FPN_Weights.DEFAULT.url
    model_filename = "fasterrcnn.pth"
    model_path = cache_path / model_filename
    if not model_path.exists():
        torch.hub.download_url_to_file(model_url, model_path, progress=False)
    model = faster_rcnn.FasterRCNN((800, 800), 3, 91)
    model.to(device)
    model.load_ots_model(model_path)

def test_fcos_ots_loader(device, cache_path):
    model_url = torchvision.models.detection.FCOS_ResNet50_FPN_Weights.DEFAULT.url
    model_filename = "fcos.pth"
    model_path = cache_path / model_filename
    if not model_path.exists():
        torch.hub.download_url_to_file(model_url, model_path, progress=False)
    model = fcos.FCOS_Full((800, 800), 3, 91)
    model.to(device)
    model.load_ots_model(model_path)

def test_mobilepose_ots_loader(device, cache_path):
    model_url = r"https://drive.google.com/uc?export=download&id=15Ihv1bVQv6_tYTFlECJMNrXEmrrka5g4"
    model_filename = "mobilepose.pth"
    model_path = cache_path / model_filename
    if not model_path.exists():
        with urllib.request.urlopen(model_url) as read_file:
            with open(model_path, 'wb') as write_file:
                write_file.write(read_file.read(int(read_file.headers["Content-Length"])))
    model = mobilepose.MobilePose_Full(3, 16)
    model.to(device)
    model.load_ots_model(model_path)
