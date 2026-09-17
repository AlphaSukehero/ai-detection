"""The uploads directory holds patient-derived images; it must not grow forever.

sweep_uploads() deletes only files matching the generated-name templates, so
anything an operator deliberately placed there survives.
"""
import os
import time

import pytest

import app as flask_app


@pytest.fixture
def uploads(tmp_path, monkeypatch):
    monkeypatch.setitem(flask_app.app.config, "UPLOAD_FOLDER", str(tmp_path))
    return tmp_path


def _touch(folder, name, age_seconds=0):
    p = folder / name
    p.write_bytes(b"x")
    if age_seconds:
        old = time.time() - age_seconds
        os.utime(p, (old, old))
    return p


def test_sweep_removes_generated_files_past_the_ttl(uploads):
    stale = _touch(uploads, "ecg_waveform_abcdef12.png", age_seconds=48 * 3600)
    assert flask_app.sweep_uploads(ttl_seconds=3600) == 1
    assert not stale.exists()


def test_sweep_keeps_recent_generated_files(uploads):
    fresh = _touch(uploads, "mri_analysis_abcdef12.png", age_seconds=60)
    assert flask_app.sweep_uploads(ttl_seconds=3600) == 0
    assert fresh.exists()


def test_sweep_never_touches_files_it_did_not_generate(uploads):
    """An operator's own file in the folder is not the sweep's business."""
    manual = _touch(uploads, "reference_scan.jpg", age_seconds=99 * 3600)
    notes = _touch(uploads, "README.txt", age_seconds=99 * 3600)
    assert flask_app.sweep_uploads(ttl_seconds=1) == 0
    assert manual.exists() and notes.exists()


def test_sweep_on_a_missing_folder_is_not_an_error(monkeypatch, tmp_path):
    monkeypatch.setitem(flask_app.app.config, "UPLOAD_FOLDER",
                        str(tmp_path / "nope"))
    assert flask_app.sweep_uploads(ttl_seconds=1) == 0


@pytest.mark.parametrize("name", [
    "ecg_waveform_abcdef12.png",
    "ecg_upload0123abcd.csv",
    "mri_upload_0123abcd.jpeg",
    "mri_heatmap_0123abcd.png",
    "mri_highlight_0123abcd.png",
    "sat_upload_0123abcd.tif",
    "sat_analysis_0123abcd.png",
])
def test_every_generated_name_template_is_servable(name):
    """A name this app writes but will not serve is a broken image on the page."""
    assert flask_app._is_generated_name(name), name


@pytest.mark.parametrize("name", [
    "app.py", "mri_cnn.keras", "../app.py", "ecg_waveform_.png",
    "ecg_waveform_ZZZZZZZZ.png", "ecg_waveform_abcdef12.py",
    "WhatsApp Image 2026-09-05 at 5.02.16 PM.jpeg",
])
def test_names_outside_the_templates_are_refused(name):
    assert not flask_app._is_generated_name(name), name
