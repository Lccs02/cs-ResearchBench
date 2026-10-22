import importlib.util
from pathlib import Path
import pytest
from researchbench.adapters.files import ingest_config


@pytest.fixture
def generated(tmp_path):
    script = Path(__file__).parents[1] / "examples/generate_synthetic.py"
    spec = importlib.util.spec_from_file_location("synthetic", script)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.generate(tmp_path / "inputs")


@pytest.fixture
def records(generated):
    return ingest_config(generated)
