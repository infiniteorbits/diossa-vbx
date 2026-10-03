import pytest
import torch


@pytest.fixture(scope="package")
def device():
    return torch.device("cuda" if torch.cuda.is_available() else "cpu")

@pytest.fixture(scope="package")
def batchsize():
    return 1

@pytest.fixture(scope="package")
def img_size():
    return (224, 224)

@pytest.fixture(scope="package")
def channels():
    return 3

@pytest.fixture(scope="class")
def inputs(batchsize, img_size, channels, device):
    inps = (torch.rand((batchsize, channels, *img_size), device=device), )
    yield inps
    del inps
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

@pytest.fixture(scope="class")
def name(model):
    return model.__class__.__name__

