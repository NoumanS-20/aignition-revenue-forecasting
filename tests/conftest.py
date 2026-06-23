import pytest
from scripts.make_sample_data import generate


@pytest.fixture
def sample_data_dir(tmp_path):
    out = tmp_path / "data"
    generate(str(out), n_days=120, seed=7)
    return str(out)
