"""Studies: one saved ECG / EEG / MRI analysis for one patient.

Append-only. A study row, its measurements and its files are saved together
or not at all.
"""
import json
from datetime import date, datetime

from records import store

MODALITIES = ("ecg", "eeg", "mri")


def _json(obj):
    def default(o):
        if hasattr(o, "item"):          # numpy scalars
            return o.item()
        if hasattr(o, "tolist"):        # numpy arrays
            return o.tolist()
        return str(o)
    return json.dumps(obj, default=default)


def save_study(conn, root, patient_id, modality, visit, result, report,
               verdict, headline, model_name, model_version, measurements,
               files, reanalysis_of=None):
    """Save one study atomically and return it (as get_study would).

    measurements: [(key, value_or_None, unit)]; files: {kind: source_path}.
    """
    if modality not in MODALITIES:
        raise ValueError(f"unknown modality {modality!r}")
    if conn.execute("SELECT 1 FROM patients WHERE id = ?", (patient_id,)).fetchone() is None:
        raise KeyError(patient_id)
    study_date = visit.get("study_date") or date.today().isoformat()
    datetime.strptime(study_date, "%Y-%m-%d")

    staging, names = store.stage(root, files)
    final = None
    conn.execute("BEGIN IMMEDIATE")
    try:
        day = datetime.now().strftime("%Y%m%d")
        n = conn.execute("SELECT COUNT(*) FROM studies WHERE id LIKE ?",
                         (f"ST-{day}-%",)).fetchone()[0] + 1
        sid = f"ST-{day}-{n:04d}"
        conn.execute(
            "INSERT INTO studies (id, patient_id, modality, study_date, created_at,"
            " referring_physician, clinical_history, verdict, headline, model_name,"
            " model_version, result_json, report_json, reanalysis_of)"
            " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (sid, patient_id, modality, study_date,
             datetime.now().isoformat(timespec="microseconds"),
             visit.get("referring_physician") or None,
             visit.get("clinical_history") or None,
             verdict, headline, model_name, model_version,
             _json(result), _json(report), reanalysis_of))
        conn.executemany(
            "INSERT INTO measurements (study_id, key, value, unit) VALUES (?, ?, ?, ?)",
            [(sid, k, None if v is None else float(v), u) for k, v, u in measurements])
        conn.executemany(
            "INSERT INTO study_files (study_id, kind, path) VALUES (?, ?, ?)",
            [(sid, kind, store.rel_path(patient_id, sid, name))
             for kind, name in names.items()])
        final = store.commit(root, staging, patient_id, sid)
        conn.execute("COMMIT")
    except Exception:
        conn.execute("ROLLBACK")
        store.discard(final or staging)
        raise
    return get_study(conn, sid)


def attach_file(conn, root, study, kind, data, ext):
    """Add a generated file (e.g. the PDF) to an existing study."""
    rel = store.add_file(root, study["patient_id"], study["id"], kind, data, ext)
    conn.execute("INSERT OR REPLACE INTO study_files (study_id, kind, path)"
                 " VALUES (?, ?, ?)", (study["id"], kind, rel))
    return rel


def get_study(conn, sid):
    row = conn.execute("SELECT * FROM studies WHERE id = ?", (sid,)).fetchone()
    if row is None:
        return None
    s = dict(row)
    s["result"] = json.loads(s.pop("result_json"))
    s["report"] = json.loads(s.pop("report_json"))
    s["measurements"] = {
        r["key"]: {"value": r["value"], "unit": r["unit"]}
        for r in conn.execute("SELECT key, value, unit FROM measurements"
                              " WHERE study_id = ?", (sid,))}
    s["files"] = {r["kind"]: r["path"] for r in conn.execute(
        "SELECT kind, path FROM study_files WHERE study_id = ?", (sid,))}
    return s


def list_studies(conn, patient_id, modality=None):
    """Summaries, newest first (study date, then time saved)."""
    sql = ("SELECT id, patient_id, modality, study_date, created_at, verdict,"
           " headline, model_name, referring_physician FROM studies"
           " WHERE patient_id = ?")
    args = [patient_id]
    if modality:
        sql += " AND modality = ?"
        args.append(modality)
    sql += " ORDER BY study_date DESC, created_at DESC"
    return [dict(r) for r in conn.execute(sql, args)]


def previous_study(conn, study):
    """The latest same-modality study for this patient before `study`."""
    row = conn.execute(
        "SELECT id FROM studies WHERE patient_id = ? AND modality = ? AND id != ?"
        " AND (study_date < ? OR (study_date = ? AND created_at < ?))"
        " ORDER BY study_date DESC, created_at DESC LIMIT 1",
        (study["patient_id"], study["modality"], study["id"],
         study["study_date"], study["study_date"], study["created_at"])).fetchone()
    return get_study(conn, row["id"]) if row else None


def measurement_series(conn, patient_id, modality):
    """{key: [(study_date, value, study_id), ...]} oldest first."""
    series = {}
    for r in conn.execute(
            "SELECT s.study_date, s.id, m.key, m.value FROM measurements m"
            " JOIN studies s ON s.id = m.study_id"
            " WHERE s.patient_id = ? AND s.modality = ?"
            " ORDER BY s.study_date, s.created_at", (patient_id, modality)):
        series.setdefault(r["key"], []).append((r["study_date"], r["value"], r["id"]))
    return series
