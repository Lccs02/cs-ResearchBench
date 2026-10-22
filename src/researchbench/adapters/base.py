"""Adapter contract: deterministic conversion only; no metric duplication."""
from pathlib import Path
from typing import Protocol
from researchbench.core.schema import ExperimentRecord


class Adapter(Protocol):
    def read(self, run: dict, *, base_dir: Path, config_source: dict) -> ExperimentRecord:
        """Return one validated record, retaining source hashes and mappings."""
        ...


class UnconfiguredProjectAdapter:
    project_name = "Unverified"

    def read(self, run: dict, *, base_dir: Path, config_source: dict) -> ExperimentRecord:
        raise NotImplementedError(
            f"{self.project_name}: real format Unverified. Supply an inspected mapping "
            "to ConfigurableFileAdapter or implement Adapter.read using confirmed samples."
        )


class PACSRAdapter(UnconfiguredProjectAdapter):
    project_name = "PACSR"


class SatelliteMultiServiceAdapter(UnconfiguredProjectAdapter):
    project_name = "Satellite multi-service forecasting"


class WorldModelRLAdapter(UnconfiguredProjectAdapter):
    project_name = "World Model RL"
