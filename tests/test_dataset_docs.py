import os

import pytest

pytestmark = pytest.mark.skipif(
    not os.path.exists("data/DATASET.md"), reason="not written yet")


def test_documents_every_domain():
    text = open("data/DATASET.md").read()
    for token in ["EuroSAT", "MIT-BIH", "inter-patient", "licence", "SHA-256"]:
        assert token.lower() in text.lower(), f"DATASET.md missing {token}"
