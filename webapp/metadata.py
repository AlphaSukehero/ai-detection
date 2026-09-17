"""The patient / survey metadata block shared by the pages and the reports.

Split out of app.py because both the Flask routes and the PDF builder need
these definitions, and having the renderer import them from the application
module made the dependency circular.
"""
import uuid
from datetime import datetime

# The string every page and report prints for a field the user left blank.
NOT_PROVIDED = "Not provided"


GENDER_OPTIONS = ["Male", "Female", "Other", "Prefer not to say"]

# Fields captured for clinical (ECG / MRI) studies.
PATIENT_FIELDS = [
    ("patient_name", "Patient Name"),
    ("patient_id", "Patient ID / MRN"),
    ("age", "Age"),
    ("gender", "Gender"),
    ("contact", "Contact Number"),
    ("referring_physician", "Referring Physician"),
    ("study_date", "Study Date"),
    ("clinical_history", "Clinical History / Indication"),
]

# Fields captured for non-clinical satellite surveys.
SURVEY_FIELDS = [
    ("site_name", "Site / Area Name"),
    ("survey_id", "Survey Reference ID"),
    ("coordinates", "Coordinates (lat, lon)"),
    ("capture_date", "Image Capture Date"),
    ("sensor", "Sensor / Source"),
    ("analyst", "Analyst"),
    ("survey_notes", "Survey Notes"),
]



def _clean_text(value, max_length=200):
    """Trim, collapse whitespace and bound the length of a free-text field."""
    if value is None:
        return ""
    text = " ".join(str(value).split())
    return text[:max_length]


def collect_metadata(form, fields):
    """Read a metadata block from a submitted form.

    Returns (values, errors). Values always contain every key so templates and
    PDF builders can rely on them; blank entries stay empty here and are filled
    in by finalize_metadata().
    """
    values = {}
    errors = []

    for key, _label in fields:
        limit = 1000 if key in ("clinical_history", "survey_notes") else 200
        values[key] = _clean_text(form.get(key), limit)

    if values.get("age"):
        try:
            age = int(float(values["age"]))
            if not 0 <= age <= 130:
                raise ValueError
            values["age"] = str(age)
        except (TypeError, ValueError):
            errors.append("Age must be a whole number between 0 and 130.")
            values["age"] = ""

    if values.get("gender") and values["gender"] not in GENDER_OPTIONS:
        errors.append("Please select a valid gender option.")
        values["gender"] = ""

    for date_key in ("study_date", "capture_date"):
        if values.get(date_key):
            try:
                datetime.strptime(values[date_key], "%Y-%m-%d")
            except ValueError:
                errors.append("Date must be in YYYY-MM-DD format.")
                values[date_key] = ""

    return values, errors


def finalize_metadata(values, fields):
    """Fill blanks with a placeholder and attach report identity fields."""
    finalized = {key: (values.get(key) or NOT_PROVIDED) for key, _ in fields}
    finalized["report_id"] = "RPT-" + uuid.uuid4().hex[:10].upper()
    finalized["generated_at"] = datetime.now().strftime("%d %b %Y, %H:%M:%S")
    return finalized


def metadata_rows(meta, fields):
    """Ordered (label, value) pairs for rendering in templates and PDFs."""
    return [(label, meta.get(key, NOT_PROVIDED)) for key, label in fields]


