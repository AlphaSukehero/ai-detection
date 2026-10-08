"""Put the repository root on sys.path for the test suite.

Without this, `pytest` and `python -m pytest` disagree: the module form puts
the working directory on sys.path and the bare console script does not, so
`import app` succeeds locally and fails in CI. Anchoring it here makes the
suite independent of how it was invoked.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

# A quick analysis with a Patient ID is saved to the record store. The suite
# must never write into the real one: app.py reads RECORDS_ROOT at import.
import tempfile

import pytest

os.environ["RECORDS_ROOT"] = tempfile.mkdtemp(prefix="records-test-")


@pytest.fixture(autouse=True)
def _throwaway_record_store(tmp_path, monkeypatch):
    """Each test gets its own empty store, so no test sees another's studies."""
    module = sys.modules.get("app")
    if module is not None:
        monkeypatch.setitem(module.app.config, "RECORDS_ROOT",
                            str(tmp_path / "records"))
    yield
