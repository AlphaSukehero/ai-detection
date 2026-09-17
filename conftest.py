"""Put the repository root on sys.path for the test suite.

Without this, `pytest` and `python -m pytest` disagree: the module form puts
the working directory on sys.path and the bare console script does not, so
`import app` succeeds locally and fails in CI. Anchoring it here makes the
suite independent of how it was invoked.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
