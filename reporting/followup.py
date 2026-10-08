"""The previous report of the same kind, carried inside a follow-up report.

A second visit is read against the first. The follow-up report therefore
shows the previous study's details, its images and what changed, so the two
do not have to be laid side by side by hand.
"""
from records.catalog import MODALITY_LABELS, entries
from reporting.history import _compare_rows


def _value(m, unit):
    if m is None or m["value"] is None:
        return "not measured"
    return f"{round(m['value'], 2):g} {unit}".strip()


def previous_details(previous):
    """[(label, value)] describing a stored study, measurements included."""
    rows = [("Study ID", previous["id"]),
            ("Study date", previous["study_date"]),
            ("Overall verdict", previous.get("verdict") or "NOT ASSESSED"),
            ("Finding", previous.get("headline") or "—"),
            ("Referring physician", previous.get("referring_physician") or "—"),
            ("Clinical history", previous.get("clinical_history") or "—")]
    for key, label, unit, _rng in entries(previous["modality"]):
        if key in previous["measurements"]:
            rows.append((label, _value(previous["measurements"][key], unit)))
    return rows


def previous_sections(previous, current, images):
    """Report sections for `previous`, to embed in the report of `current`.

    images: [(caption, path)] of the previous study.
    """
    label = MODALITY_LABELS[previous["modality"]]
    when = previous["study_date"]
    sections = [("table", f"Previous {label} Report — {previous['id']} ({when})",
                 [("Field", "Details")] + previous_details(previous))]
    sections += [("image", f"Previous {caption} — {when}", path)
                 for caption, path in images]
    sections.append(("table", "Change Since Previous Report",
                     _compare_rows(previous, current)))
    return sections
