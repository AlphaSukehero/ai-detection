"""Patient registry: register once, find again by ID, MRN, name or phone."""
from datetime import date, datetime

from records.db import transaction

FIELDS = ("name", "dob", "sex", "phone", "address", "notes")
LIMITS = {"name": 200, "dob": 10, "sex": 40, "phone": 40, "address": 500,
          "notes": 2000}


def _clean(value, limit):
    text = " ".join(str(value or "").split())
    return text[:limit] or None


def _parse_date(value):
    return datetime.strptime(value, "%Y-%m-%d").date()


def register(conn, **fields):
    """Create a patient and return it. The portal issues the ID."""
    values = {k: _clean(fields.get(k), LIMITS[k]) for k in FIELDS}
    if not values["name"]:
        raise ValueError("Patient name is required.")
    if values["dob"]:
        try:
            dob = _parse_date(values["dob"])
        except ValueError as e:
            raise ValueError("Date of birth must be YYYY-MM-DD.") from e
        if dob > date.today():
            raise ValueError("Date of birth cannot be in the future.")
    with transaction(conn):
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM patients").fetchone()[0]
        pid = f"PT-{seq:06d}"
        conn.execute(
            "INSERT INTO patients (seq, id, name, dob, sex, phone, address, notes,"
            " created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (seq, pid, *(values[k] for k in FIELDS),
             datetime.now().isoformat(timespec="seconds")))
    return get(conn, pid)


def clean_id(value):
    """A typed patient ID as it is stored: trimmed, inner spaces collapsed."""
    return _clean(value, 200)


def find_by_typed_id(conn, typed):
    """The patient a typed ID names: the portal's ID or the clinic's MRN."""
    key = (clean_id(typed) or "").lower()
    if not key:
        return None
    row = conn.execute("SELECT * FROM patients WHERE lower(id) = ? OR lower(mrn) = ?"
                       " ORDER BY lower(id) = ? DESC LIMIT 1", (key, key, key)).fetchone()
    return _as_dict(row) if row else None


def find_or_create(conn, typed_id, name=None, sex=None, phone=None):
    """(patient, created) for the ID typed on an analysis form.

    The same ID is the same patient, whatever else was typed: that is what
    lets a second visit find the first. A new ID opens a record under it.
    """
    mrn = clean_id(typed_id)
    if not mrn:
        raise ValueError("A patient ID is required to keep a record.")
    found = find_by_typed_id(conn, mrn)
    if found:
        return found, False
    with transaction(conn):
        seq = conn.execute("SELECT COALESCE(MAX(seq), 0) + 1 FROM patients").fetchone()[0]
        pid = f"PT-{seq:06d}"
        conn.execute(
            "INSERT INTO patients (seq, id, name, sex, phone, mrn, created_at)"
            " VALUES (?, ?, ?, ?, ?, ?, ?)",
            (seq, pid, _clean(name, LIMITS["name"]) or f"Patient {mrn}",
             _clean(sex, LIMITS["sex"]), _clean(phone, LIMITS["phone"]), mrn,
             datetime.now().isoformat(timespec="seconds")))
    return get(conn, pid), True


def get(conn, pid):
    row = conn.execute("SELECT * FROM patients WHERE id = ?", (pid,)).fetchone()
    return _as_dict(row) if row else None


def search(conn, query, limit=50):
    """Match ID, name (case-insensitive substring) or phone digits."""
    q = " ".join(str(query or "").split())
    if not q:
        rows = conn.execute("SELECT * FROM patients ORDER BY seq DESC LIMIT ?",
                            (limit,)).fetchall()
        return [_as_dict(r) for r in rows]
    digits = "".join(c for c in q if c.isdigit())
    like = f"%{q.lower()}%"
    rows = conn.execute(
        "SELECT * FROM patients WHERE lower(id) LIKE ? OR lower(name) LIKE ?"
        " OR lower(COALESCE(mrn, '')) LIKE ?"
        " OR (? != '' AND replace(replace(replace(COALESCE(phone, ''), ' ', ''),"
        " '-', ''), '+', '') LIKE ?)"
        " ORDER BY seq DESC LIMIT ?",
        (like, like, like, digits, f"%{digits}%", limit)).fetchall()
    return [_as_dict(r) for r in rows]


def age_on(dob, on_date):
    """Age in whole years on a given date, or None if DOB is unknown."""
    if not dob or not on_date:
        return None
    b, d = _parse_date(dob), _parse_date(on_date)
    return d.year - b.year - ((d.month, d.day) < (b.month, b.day))


def _as_dict(row):
    p = dict(row)
    p.pop("seq", None)
    p["age"] = age_on(p.get("dob"), date.today().isoformat())
    return p
