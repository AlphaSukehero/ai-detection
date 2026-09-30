"""Shared fixture: a Flask client bound to a throwaway record store."""
import pytest

import app as flask_app


@pytest.fixture
def rclient(tmp_path, monkeypatch):
    monkeypatch.setitem(flask_app.app.config, "RECORDS_ROOT", str(tmp_path / "rec"))
    flask_app.app.config["TESTING"] = True
    with flask_app.app.test_client() as c:
        yield c


def register(client, **fields):
    data = {"name": "Asha Rao", "dob": "1980-02-01", "sex": "Female",
            "phone": "98450 12345", "address": "", "notes": "Penicillin allergy"}
    data.update(fields)
    resp = client.post("/patients/new", data=data)
    assert resp.status_code == 302, resp.get_data(as_text=True)[:500]
    return resp.headers["Location"].rstrip("/").split("/")[-1]
