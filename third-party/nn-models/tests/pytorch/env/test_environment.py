import pytest
import torch


@pytest.fixture(params=[(1, 2, 3), (1.2, 2.3)])
def data(request):
    return request.param


def test_cpu(data):
    l = list(data)
    t = torch.as_tensor(data, device="cpu")
    assert l == pytest.approx(t.numpy().tolist())

@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA is not available...")
def test_cuda(data):
    l = list(data)
    t = torch.as_tensor(data, device="cuda")
    assert l == pytest.approx(t.cpu().numpy().tolist())

