"""The patient record store: registry, studies, files, comparison."""
import os

import pytest

from records import db, patients, studies
from records.compare import compare


@pytest.fixture
def root(tmp_path):
    return str(tmp_path / "records")


@pytest.fixture
def conn(root):
    c = db.connect(root)
    yield c
    c.close()


def _src(tmp_path, name, content=b"x"):
    p = tmp_path / name
    p.write_bytes(content)
    return str(p)


def _save(conn, root, pid, tmp_path, modality="ecg", date="2026-09-01",
          measurements=None, verdict="NORMAL", headline="ok", files=None):
    return studies.save_study(
        conn, root, pid, modality,
        visit={"study_date": date, "referring_physician": "Dr A",
               "clinical_history": "palpitations"},
        result={"prediction": headline},
        report={"k": 1},
        verdict=verdict, headline=headline,
        model_name="ecg_cnn", model_version="2026-09-17",
        measurements=measurements or [("heart_rate", 72.0, "bpm")],
        files=files if files is not None else {"original": _src(tmp_path, "a.csv")})


# ------------------------------------------------------------ patients

def test_register_issues_sequential_ids(conn):
    a = patients.register(conn, name="Asha Rao", dob="1980-02-01", sex="Female")
    b = patients.register(conn, name="Ben Ode")
    assert (a["id"], b["id"]) == ("PT-000001", "PT-000002")


def test_register_requires_a_name(conn):
    with pytest.raises(ValueError):
        patients.register(conn, name="  ")


def test_register_rejects_a_bad_dob(conn):
    with pytest.raises(ValueError):
        patients.register(conn, name="A", dob="01/02/1980")


def test_ids_survive_reconnect(root):
    c = db.connect(root)
    patients.register(c, name="A")
    c.close()
    c = db.connect(root)
    assert patients.register(c, name="B")["id"] == "PT-000002"
    c.close()


def test_search_by_id_name_and_phone(conn):
    a = patients.register(conn, name="Asha Rao", phone="98450 12345")
    patients.register(conn, name="Ben Ode", phone="11111")
    assert [p["id"] for p in patients.search(conn, "PT-000001")] == [a["id"]]
    assert [p["id"] for p in patients.search(conn, "asha")] == [a["id"]]
    assert [p["id"] for p in patients.search(conn, "12345")] == [a["id"]]
    assert len(patients.search(conn, "")) == 2


def test_get_unknown_patient_is_none(conn):
    assert patients.get(conn, "PT-999999") is None


def test_age_is_computed_on_the_study_date():
    assert patients.age_on("1980-06-15", "2026-06-14") == 45
    assert patients.age_on("1980-06-15", "2026-06-15") == 46
    assert patients.age_on(None, "2026-06-15") is None


# ------------------------------------------------------------- studies

def test_save_study_stores_row_measurements_and_files(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    s = _save(conn, root, p["id"], tmp_path,
              files={"original": _src(tmp_path, "a.csv", b"data"),
                     "waveform": _src(tmp_path, "w.png", b"png")})
    assert s["id"].startswith("ST-") and s["patient_id"] == p["id"]
    got = studies.get_study(conn, s["id"])
    assert got["measurements"]["heart_rate"]["value"] == 72.0
    assert got["result"] == {"prediction": "ok"}
    assert got["report"] == {"k": 1}
    for kind, content in (("original", b"data"), ("waveform", b"png")):
        path = os.path.join(root, got["files"][kind])
        assert open(path, "rb").read() == content


def test_unmeasured_values_are_null_not_zero(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    s = _save(conn, root, p["id"], tmp_path,
              measurements=[("heart_rate", None, "bpm")])
    assert studies.get_study(conn, s["id"])["measurements"]["heart_rate"]["value"] is None


def test_failed_save_leaves_nothing_behind(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    with pytest.raises(FileNotFoundError):
        _save(conn, root, p["id"], tmp_path,
              files={"original": str(tmp_path / "missing.csv")})
    assert studies.list_studies(conn, p["id"]) == []
    pdir = os.path.join(root, "files", p["id"])
    assert not os.path.exists(pdir) or os.listdir(pdir) == []


def test_save_for_unknown_patient_is_refused(conn, root, tmp_path):
    with pytest.raises(KeyError):
        _save(conn, root, "PT-424242", tmp_path)


def test_list_is_newest_first_and_filterable(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    old = _save(conn, root, p["id"], tmp_path, date="2026-01-01")
    new = _save(conn, root, p["id"], tmp_path, date="2026-05-01")
    eeg = _save(conn, root, p["id"], tmp_path, modality="eeg", date="2026-03-01")
    assert [s["id"] for s in studies.list_studies(conn, p["id"])] == \
        [new["id"], eeg["id"], old["id"]]
    assert [s["id"] for s in studies.list_studies(conn, p["id"], "ecg")] == \
        [new["id"], old["id"]]


def test_previous_study_is_the_latest_earlier_one_of_that_modality(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    first = _save(conn, root, p["id"], tmp_path, date="2026-01-01")
    _save(conn, root, p["id"], tmp_path, modality="eeg", date="2026-02-01")
    second = _save(conn, root, p["id"], tmp_path, date="2026-03-01")
    assert studies.previous_study(conn, second)["id"] == first["id"]
    assert studies.previous_study(conn, first) is None


def test_measurement_series_is_chronological(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    _save(conn, root, p["id"], tmp_path, date="2026-03-01",
          measurements=[("heart_rate", 90.0, "bpm")])
    _save(conn, root, p["id"], tmp_path, date="2026-01-01",
          measurements=[("heart_rate", 70.0, "bpm")])
    series = studies.measurement_series(conn, p["id"], "ecg")
    assert [v for _d, v, _sid in series["heart_rate"]] == [70.0, 90.0]


# ------------------------------------------------------------- compare

def test_compare_reports_delta_direction_and_range_change(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    a = _save(conn, root, p["id"], tmp_path, date="2026-01-01",
              measurements=[("qtc", 430.0, "ms"), ("heart_rate", 72.0, "bpm")])
    b = _save(conn, root, p["id"], tmp_path, date="2026-02-01",
              measurements=[("qtc", 482.0, "ms"), ("heart_rate", None, "bpm")],
              verdict="ABNORMAL", headline="Prolonged QTc")
    rows = {r["key"]: r for r in compare(studies.get_study(conn, a["id"]),
                                         studies.get_study(conn, b["id"]))}
    assert rows["qtc"]["delta"] == pytest.approx(52.0)
    assert rows["qtc"]["direction"] == "up"
    assert "above normal" in rows["qtc"]["note"]
    assert rows["heart_rate"]["delta"] is None
    assert rows["verdict"]["before"] == "NORMAL" and rows["verdict"]["after"] == "ABNORMAL"
    assert rows["verdict"]["changed"] is True


def test_compare_refuses_different_modalities(conn, root, tmp_path):
    p = patients.register(conn, name="A")
    a = _save(conn, root, p["id"], tmp_path)
    b = _save(conn, root, p["id"], tmp_path, modality="eeg")
    with pytest.raises(ValueError):
        compare(studies.get_study(conn, a["id"]), studies.get_study(conn, b["id"]))


def test_db_failure_after_staging_leaves_no_files(conn, root, tmp_path):
    import sqlite3
    p = patients.register(conn, name="A")
    with pytest.raises(sqlite3.IntegrityError):
        _save(conn, root, p["id"], tmp_path,
              measurements=[("qtc", 1.0, "ms"), ("qtc", 2.0, "ms")])
    assert studies.list_studies(conn, p["id"]) == []
    left = [f for _d, _s, fs in os.walk(os.path.join(root, "files")) for f in fs]
    assert left == []
