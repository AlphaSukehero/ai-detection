"""Patient registry pages: register, search, patient timeline, stored files.

One portal, no login (see docs/superpowers/specs/2026-09-30-patient-records-design.md).
The record store lives at app.config["RECORDS_ROOT"].
"""
import os

from flask import (Blueprint, abort, current_app, g, redirect, render_template,
                   request, send_file, url_for)

from records import db, patients, studies, store
from records.catalog import MODALITY_LABELS

bp = Blueprint("patients", __name__)

MODALITY_PAGES = {"ecg": "/ecg", "eeg": "/eeg", "mri": "/brain-tumor"}
SEX_OPTIONS = ["Male", "Female", "Other"]


def records_root():
    return current_app.config["RECORDS_ROOT"]


def get_conn():
    """One connection per request, to the store configured right now."""
    root = records_root()
    if g.get("records_conn") is None or g.get("records_conn_root") != root:
        g.records_conn = db.connect(root)
        g.records_conn_root = root
    return g.records_conn


@bp.teardown_app_request
def _close(_exc):
    conn = g.pop("records_conn", None)
    if conn is not None:
        conn.close()


def patient_or_404(pid):
    p = patients.get(get_conn(), pid)
    if p is None:
        abort(404, description=f"No patient with ID {pid}.")
    return p


def study_or_404(pid, sid):
    s = studies.get_study(get_conn(), sid)
    if s is None or s["patient_id"] != pid:
        abort(404, description=f"No study {sid} for patient {pid}.")
    return s


@bp.app_context_processor
def _labels():
    return {"modality_labels": MODALITY_LABELS, "modality_pages": MODALITY_PAGES}


@bp.route("/patients")
def search():
    q = request.args.get("q", "")
    return render_template("patients/search.html", q=q,
                           results=patients.search(get_conn(), q))


@bp.route("/patients/new", methods=["GET", "POST"])
def register():
    form = request.form if request.method == "POST" else {}
    if request.method == "POST":
        try:
            p = patients.register(get_conn(), **{k: form.get(k) for k in patients.FIELDS})
        except ValueError as e:
            return render_template("patients/register.html", error=str(e),
                                   form=form, sex_options=SEX_OPTIONS)
        return redirect(url_for("patients.detail", pid=p["id"]))
    return render_template("patients/register.html", error=None, form=form,
                           sex_options=SEX_OPTIONS)


@bp.route("/patients/<pid>")
def detail(pid):
    p = patient_or_404(pid)
    conn = get_conn()
    history = studies.list_studies(conn, pid)
    by_modality = {}
    for s in history:
        by_modality.setdefault(s["modality"], []).append(s)
    return render_template("patients/detail.html", patient=p, studies=history,
                           by_modality=by_modality)


@bp.route("/patients/<pid>/studies/<sid>/files/<kind>")
def study_file(pid, sid, kind):
    s = study_or_404(pid, sid)
    rel = s["files"].get(kind)
    if rel is None:
        abort(404)
    path = store.absolute(records_root(), rel)
    if not os.path.isfile(path):
        abort(404, description="This stored file is missing from the record store.")
    return send_file(path, as_attachment=request.args.get("download") == "1")


@bp.app_errorhandler(404)
def _not_found(e):
    return render_template("patients/not_found.html",
                           message=getattr(e, "description", "Not found.")), 404
