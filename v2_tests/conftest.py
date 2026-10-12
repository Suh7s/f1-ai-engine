import shutil
import pytest
from f1engine.demo import load_demo, DEMO_EVENT, DEMO_CUTOFF
from f1engine.store import Store
from f1engine.contracts import ForecastRequest


@pytest.fixture(scope="session")
def template(tmp_path_factory):
    root = tmp_path_factory.mktemp("fixture")
    load_demo(Store(root), drivers=12, rounds=8)
    return root


@pytest.fixture
def store(template, tmp_path):
    shutil.copytree(template, tmp_path / "data")
    return Store(tmp_path / "data")


@pytest.fixture
def request_():
    return ForecastRequest(event_id=DEMO_EVENT, as_of_utc=DEMO_CUTOFF, stage="post_sprint", simulations=1000)
