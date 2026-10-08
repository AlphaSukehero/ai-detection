"""Patient registry pages: register, search, patient timeline, stored files.

One portal, no login (see docs/superpowers/specs/2026-09-30-patient-records-design.md).
The record store lives at app.config["RECORDS_ROOT"].
"""
import os

import io

from flask import (Blueprint, Response, abort, current_app, g, redirect,
                   render_template, request, send_file, url_for)

from records import db, patients, studies, store
from records.catalog import MODALITY_LABELS, entries, range_status
from records.compare import compare as compare_studies

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
    # Default comparison per modality: the two most recent studies.
    latest_pairs = {m: (ss[1]["id"], ss[0]["id"]) for m, ss in by_modality.items()
                    if len(ss) >= 2}
    return render_template("patients/detail.html", patient=p, studies=history,
                           by_modality=by_modality, latest_pairs=latest_pairs)


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


IMAGE_KINDS = [("scan", "MRI scan"), ("heatmap", "Grad-CAM attention map"),
               ("highlight", "Highest-attention region"),
               ("waveform", "ECG waveform"), ("timeline", "EEG timeline")]


def measurement_rows(study):
    """Catalog-ordered rows for a study's measurements, with range status."""
    rows = []
    for key, label, unit, rng in entries(study["modality"]):
        m = study["measurements"].get(key)
        if m is None:
            continue
        value = m["value"]
        rows.append({"label": label, "unit": unit,
                     "value": None if value is None else round(value, 2),
                     "range": rng, "status": range_status(value, rng)})
    return rows


@bp.route("/patients/<pid>/studies/<sid>")
def study(pid, sid):
    p = patient_or_404(pid)
    s = study_or_404(pid, sid)
    previous = studies.previous_study(get_conn(), s)
    images = [(kind, caption) for kind, caption in IMAGE_KINDS if kind in s["files"]]
    return render_template("patients/study.html", patient=p, study=s,
                           previous=previous, rows=measurement_rows(s),
                           images=images)


@bp.route("/patients/<pid>/studies/<sid>/pdf")
def study_pdf(pid, sid):
    s = study_or_404(pid, sid)
    rel = s["files"].get("pdf")
    path = store.absolute(records_root(), rel) if rel else None
    if path and os.path.isfile(path):
        return send_file(path, mimetype="application/pdf", as_attachment=True,
                         download_name=f"{sid}_{MODALITY_LABELS[s['modality']].replace(' ', '_')}.pdf")
    # Stored PDF missing: rebuild it from the stored report rather than fail.
    from webapp.study_binding import render_stored_pdf
    buffer, _name = render_stored_pdf(
        records_root(), s, studies.previous_study(get_conn(), s))
    return send_file(buffer, mimetype="application/pdf", as_attachment=True,
                     download_name=f"{sid}.pdf")


@bp.route("/patients/<pid>/compare")
def compare(pid):
    p = patient_or_404(pid)
    a_id, b_id = request.args.get("a"), request.args.get("b")
    history = studies.list_studies(get_conn(), pid)
    if not (a_id and b_id):
        return render_template("patients/compare.html", patient=p, pair=None,
                               studies=history, rows=None, error=None)
    a, b = study_or_404(pid, a_id), study_or_404(pid, b_id)
    if (a["study_date"], a["created_at"]) > (b["study_date"], b["created_at"]):
        a, b = b, a
    try:
        rows = compare_studies(a, b)
    except ValueError as e:
        return render_template("patients/compare.html", patient=p, pair=None,
                               studies=history, rows=None, error=str(e)), 400
    images = [(kind, caption) for kind, caption in IMAGE_KINDS
              if kind in a["files"] or kind in b["files"]]
    return render_template("patients/compare.html", patient=p, pair=(a, b),
                           studies=history, rows=rows, images=images, error=None)


@bp.route("/patients/<pid>/trends/<modality>.png")
def trend(pid, modality):
    patient_or_404(pid)
    if modality not in MODALITY_LABELS:
        abort(404)
    from reporting.trends import trend_png
    png = trend_png(modality, studies.measurement_series(get_conn(), pid, modality))
    if png is None:
        abort(404, description="No measurements to chart for this modality.")
    return Response(png, mimetype="image/png")


@bp.route("/patients/<pid>/history.pdf")
def history_pdf(pid):
    p = patient_or_404(pid)
    conn = get_conn()
    full = [studies.get_study(conn, s["id"]) for s in studies.list_studies(conn, pid)]
    from reporting.history import build_history_pdf
    from reporting.trends import trend_png
    from webapp.study_binding import study_images
    trends = {}
    for modality in MODALITY_LABELS:
        png = trend_png(modality, studies.measurement_series(conn, pid, modality), dpi=150)
        if png:
            trends[modality] = png
    buffer = build_history_pdf(p, full, trends,
                               lambda s: study_images(records_root(), s))
    return send_file(io.BytesIO(buffer.getvalue()), mimetype="application/pdf",
                     as_attachment=True,
                     download_name=f"Patient_History_{p['id']}.pdf")


@bp.app_errorhandler(404)
def _not_found(e):
    return render_template("patients/not_found.html",
                           message=getattr(e, "description", "Not found.")), 404
