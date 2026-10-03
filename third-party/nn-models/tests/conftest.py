import pytest


@pytest.fixture(scope="session")
def debug(pytestconfig):
    return pytestconfig.getoption("debug_tests", default=False)

@pytest.fixture(scope="session")
def vitis_ai(pytestconfig):
    return pytestconfig.getoption("vitis_ai", default=False)


def pytest_addoption(parser):
    parser.addoption("--debug-tests", action="store_true", help="nn-models switch: enable extra debugging features of the tests")
    parser.addoption("--vitis-ai", action="store_true", help="nn-models switch: test Vitis-AI TensorFlow 2 (Keras) quantizer API")

