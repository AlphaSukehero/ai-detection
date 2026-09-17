"""Structured clinical presentation: formula, value, reference range, verdict.

This module adds no measurement. It takes the Measurements that
ecg.parameters.analyse() already produced and dresses each one with the
formula used, the accepted reference range, and whether the value falls
inside it. Keeping it separate from parameters.py is deliberate: the
measurement layer must stay free of clinical interpretation, so that a
change to a reference range can never quietly change a measured number.

Two rules this file exists to enforce:

  * A parameter that could not be measured gets no verdict. It reports
    "Not measurable" and the reason, never a range check against None.
  * The formula reported is the formula actually used. QTc here is
    Fridericia, because that is what qtc_fridericia() computes; Bazett is
    shown alongside only when the RR interval needed to compute it is
    itself available.
"""
from ecg.quality import OK

# (key, label, formula, unit, low, high, range_text)
# Ranges are the conventional adult values; they are reference ranges, not
# diagnostic thresholds, and the verdict below says only in/out of range.
PARAMETER_SPECS = [
    ("heart_rate", "Heart Rate (HR)",
     "60 / mean RR (s) — equivalent to 300 / large boxes, 1500 / small boxes",
     "bpm", 60.0, 100.0, "60–100 bpm"),
    ("pr_interval", "PR Interval",
     "Median P-wave onset to QRS onset across beats",
     "ms", 120.0, 200.0, "0.12–0.20 s (3–5 small boxes)"),
    ("qrs_duration", "QRS Complex Duration",
     "Median QRS onset to QRS offset across beats",
     "ms", 60.0, 110.0, "0.06–0.11 s (< 3 small boxes)"),
    ("qtc", "Corrected QT Interval (QTc)",
     "Fridericia: QTc = QT / RR^(1/3)",
     "ms", None, 440.0, "Male ≤ 0.44 s | Female ≤ 0.46 s"),
    ("st_segment", "ST Segment Deviation",
     "Vertical displacement at J+60 ms relative to the PR isoelectric baseline",
     "mm", -1.0, 1.0, "0 mm (isoelectric)"),
]

# QTc ceiling by sex; the specs table carries the male value as the default.
QTC_CEILING_MS = {"male": 440.0, "female": 460.0}


def _verdict(value, low, high):
    if value is None:
        return None
    if low is not None and value < low:
        return "Below range"
    if high is not None and value > high:
        return "Above range"
    return "Within range"


def _raw(report, key):
    """The measured scalar in display units, or None if not measurable."""
    m = report.get(key)
    if m is None or m.value is None or m.quality is not OK:
        return None
    if key in ("pr_interval", "qrs_duration", "qt_interval", "qtc"):
        return m.value * 1000.0          # seconds -> ms
    return m.value


def parameter_table(report, sex=None):
    """One row per template parameter: formula, value, range, verdict."""
    ceiling = QTC_CEILING_MS.get((sex or "").strip().lower())
    rows = []
    for key, label, formula, unit, low, high, range_text in PARAMETER_SPECS:
        if key == "qtc" and ceiling is not None:
            high = ceiling
            range_text = f"≤ {ceiling / 1000:.2f} s ({sex.lower()})"
        value = _raw(report, key)
        m = report.get(key)
        if value is None:
            shown = "Not measurable"
            reason = (m.reason if m is not None and m.reason else
                      "Parameter could not be derived from this signal")
        else:
            shown = f"{value:.1f} {unit}" if unit != "bpm" else f"{value:.0f} {unit}"
            reason = None
        rows.append({
            "key": key, "label": label, "formula": formula,
            "value": shown, "reason": reason,
            "range": range_text, "verdict": _verdict(value, low, high),
        })
    return rows


def diagnostic_status(report, rows, beat_result=None):
    """Classification plus the findings that justify it.

    Returns "INDETERMINATE" rather than "NORMAL" when too little was
    measurable to support a normal call. Calling an unreadable strip normal
    is the most dangerous output this page could produce.
    """
    measurable = [r for r in rows if r["verdict"] is not None]
    out_of_range = [r for r in measurable if r["verdict"] != "Within range"]

    if len(measurable) < 3:
        return {
            "classification": "INDETERMINATE",
            "abnormality": "Insufficient measurable parameters for a reading",
            "findings": [f"{r['label']}: {r['reason']}"
                         for r in rows if r["verdict"] is None],
        }

    findings = [f"{r['label']} {r['value']} — {r['verdict'].lower()} "
                f"(normal {r['range']})" for r in out_of_range]

    rhythm = report.get("rhythm")
    irregular = rhythm is not None and rhythm.reason == "Irregular"
    if irregular:
        findings.append("RR intervals irregular (coefficient of variation "
                        "above 10%)")

    if beat_result and beat_result.get("prediction") == "ABNORMAL":
        findings.append(f"Beat classifier: {beat_result['abnormal_type']}")

    if not findings:
        return {"classification": "NORMAL",
                "abnormality": "Normal sinus rhythm",
                "findings": []}

    return {"classification": "ABNORMAL",
            "abnormality": _name_abnormality(rows, irregular),
            "findings": findings}


def _name_abnormality(rows, irregular):
    """Name only what a single-parameter deviation actually supports.

    Deliberately conservative. A short PR does not establish pre-excitation
    and a wide QRS does not establish a named bundle branch block without
    morphology this pipeline does not measure, so those are described rather
    than diagnosed. Naming a specific condition the measurements cannot
    support would be a fabricated diagnosis.
    """
    by_key = {r["key"]: r for r in rows}
    named = []

    hr = by_key.get("heart_rate", {})
    if hr.get("verdict") == "Below range":
        named.append("Bradycardia" if not irregular
                     else "Bradycardia with irregular rhythm")
    elif hr.get("verdict") == "Above range":
        named.append("Tachycardia" if not irregular
                     else "Tachycardia with irregular rhythm")
    elif irregular:
        named.append("Irregular rhythm")

    if by_key.get("pr_interval", {}).get("verdict") == "Above range":
        named.append("Prolonged PR interval (first-degree AV block pattern)")
    if by_key.get("qrs_duration", {}).get("verdict") == "Above range":
        named.append("Wide QRS complex (morphology needed to classify)")
    if by_key.get("qtc", {}).get("verdict") == "Above range":
        named.append("Prolonged QTc")
    st = by_key.get("st_segment", {})
    if st.get("verdict") == "Above range":
        named.append("ST elevation (requires urgent clinical correlation)")
    elif st.get("verdict") == "Below range":
        named.append("ST depression")

    return "; ".join(named) if named else "Abnormal parameters — see findings"


# Precautions are keyed to what was actually found. Generic advice attached
# to a specific abnormality reads as clinical guidance it is not.
PRECAUTIONS = {
    "ST elevation": [
        "ST elevation can indicate acute myocardial infarction. If this "
        "reading belongs to a symptomatic patient, treat it as an emergency "
        "and seek immediate care — do not wait for a repeat tracing.",
        "Red flags: chest pain or pressure, pain radiating to jaw or arm, "
        "breathlessness, sweating, nausea, light-headedness.",
    ],
    "Prolonged QTc": [
        "Prolonged QTc raises the risk of torsades de pointes.",
        "Review medications known to prolong QT and check potassium, "
        "magnesium and calcium.",
        "Red flags: palpitations, fainting or near-fainting, seizures.",
    ],
    "Bradycardia": [
        "Red flags: fainting, dizziness on standing, breathlessness, chest "
        "discomfort, confusion or unusual fatigue.",
        "Review rate-slowing medications (beta blockers, calcium channel "
        "blockers, digoxin) with the prescriber.",
    ],
    "Tachycardia": [
        "Red flags: chest pain, breathlessness at rest, fainting, or a "
        "sustained rate that does not settle with rest.",
        "Reversible contributors worth excluding: fever, dehydration, pain, "
        "anaemia, thyroid dysfunction, stimulants.",
    ],
    "Irregular rhythm": [
        "An irregular rhythm may indicate atrial fibrillation, which carries "
        "stroke risk and warrants formal assessment and rate/rhythm review.",
        "Red flags: sudden weakness or numbness on one side, facial droop, "
        "speech difficulty — call emergency services immediately.",
    ],
    "Prolonged PR interval": [
        "First-degree AV block is often benign but should be interpreted "
        "alongside symptoms and medications.",
        "Red flags: fainting, pre-syncope, or a progressively slowing pulse.",
    ],
    "Wide QRS complex": [
        "A wide QRS requires 12-lead morphology to classify; this pipeline "
        "measures width only.",
        "Red flags: syncope, chest pain, or breathlessness.",
    ],
}

NORMAL_ADVICE = [
    "No parameter fell outside its reference range on this tracing. A single "
    "normal ECG does not exclude intermittent arrhythmia or structural "
    "disease, and does not replace clinical assessment.",
    "General cardiovascular maintenance: regular aerobic activity, blood "
    "pressure and lipid monitoring appropriate to age and risk, no smoking, "
    "and prompt review of any new chest pain, palpitations or syncope.",
]

INDETERMINATE_ADVICE = [
    "Too few parameters were measurable to issue a reading. This reflects "
    "signal or image quality, not the absence of disease.",
    "Repeat with a clean 12-lead recording, or a strip image that includes "
    "the calibration grid so a timebase can be established.",
]

DISCLAIMER = (
    "This reading is generated by an automated screening tool, not a medical "
    "device, and is not a diagnosis. Every value and conclusion above must be "
    "cross-verified by a licensed healthcare professional against the full "
    "clinical picture before any decision is made. If the patient is "
    "symptomatic, seek care immediately regardless of what this report says."
)


def precautions(status):
    if status["classification"] == "NORMAL":
        return NORMAL_ADVICE
    if status["classification"] == "INDETERMINATE":
        return INDETERMINATE_ADVICE
    out, name = [], status["abnormality"]
    for key, advice in PRECAUTIONS.items():
        if key in name:
            out.extend(advice)
    if not out:
        out.append("Abnormal parameters were found; clinical correlation and "
                   "review by a cardiologist are required.")
    out.append("Seek emergency care for chest pain, severe breathlessness, "
               "fainting, or a sudden change in heart rhythm.")
    return out


def clinical_report(report, sex=None, beat_result=None):
    """The full structured reading: parameters, status, precautions."""
    rows = parameter_table(report, sex=sex)
    status = diagnostic_status(report, rows, beat_result)
    return {"parameters": rows, "status": status,
            "precautions": precautions(status), "disclaimer": DISCLAIMER}
