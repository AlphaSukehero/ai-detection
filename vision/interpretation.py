"""Doctor-style wording for the brain MRI report.

Mirrors eeg.interpretation: a verdict, a clinical impression, recommendations
and precautions, shared by the web page and the PDF. The precautions are
standard patient-safety guidance for the suspected lesion type, framed for
the reviewing radiologist or neurosurgeon to confirm -- the underlying call
is a four-class image classifier's, not a histological diagnosis.
"""

TUMOUR_DESCRIPTIONS = {
    "Glioma": "an intra-axial lesion with imaging features suggestive of glioma",
    "Meningioma": "an extra-axial, dural-based lesion with imaging features "
                  "suggestive of meningioma",
    "Pituitary": "a sellar / suprasellar lesion with imaging features "
                 "suggestive of pituitary adenoma",
}

GENERAL_PRECAUTION = (
    "This report is a screening aid and must be verified by a qualified "
    "radiologist before any clinical decision."
)

RED_FLAGS = (
    "Seek emergency care for sudden severe headache, repeated vomiting, a "
    "first seizure, new weakness, speech difficulty, confusion or loss of "
    "consciousness."
)


def mri_verdict(prediction):
    """Overall NORMAL / ABNORMAL call. No prediction means NOT ASSESSED."""
    prediction = (prediction or "").strip()
    if prediction != "No Tumor" and prediction not in TUMOUR_DESCRIPTIONS:
        return {"label": "NOT ASSESSED", "tone": "warn",
                "detail": "No classification was produced for this study."}
    if prediction == "No Tumor":
        return {"label": "NORMAL", "tone": "good",
                "detail": "No tumour detected by the classifier."}
    return {"label": "ABNORMAL", "tone": "bad",
            "detail": f"{prediction} suspected."}


def _given(values, key):
    value = str(values.get(key) or "").strip()
    return value if value and value != "Not provided" else None


def _impression(prediction, v):
    confidence = _given(v, "confidence")
    conf_text = f" (classifier confidence {confidence}%)" if confidence else ""

    if prediction == "No Tumor":
        return ("No focal mass or structural lesion was identified by the "
                f"classifier{conf_text}. This screening result does not "
                "exclude small, early or non-enhancing lesions, or pathology "
                "outside the four trained classes; correlate with symptoms "
                "and the full radiological read.")

    if prediction not in TUMOUR_DESCRIPTIONS:
        return ("No automated classification is available for this study. "
                "The images require full review by a radiologist; this "
                "report offers no reassurance.")

    location = _given(v, "location")
    area = _given(v, "area")
    severity = _given(v, "severity")
    detail = []
    if location:
        where = location if location.lower().endswith(("region", "area")) \
            else f"{location} region"
        detail.append(f"the highest model activation lies in the {where}")
    if area:
        detail.append(f"covering approximately {area}% of the image")
    if severity:
        detail.append(f"with a {severity.lower()} estimated severity")
    detail_text = (", " + ", ".join(detail)) if detail else ""
    return (f"ABNORMAL study. The MRI shows "
            f"{TUMOUR_DESCRIPTIONS[prediction]}{conf_text}{detail_text}. "
            "Localisation and size are derived from activation maps, not "
            "calibrated measurement. Histological confirmation and "
            "specialist review are required before any diagnosis.")


def _recommendations(prediction):
    if prediction == "Glioma":
        return [
            "Urgent neurosurgical / neuro-oncology referral.",
            "Contrast-enhanced MRI brain with perfusion and spectroscopy if "
            "available, to characterise grade.",
            "Discussion at a multidisciplinary tumour board; tissue diagnosis "
            "(biopsy or resection) for histology and molecular markers.",
            "Assess for raised intracranial pressure and seizure activity.",
        ]
    if prediction == "Meningioma":
        return [
            "Neurosurgical assessment of size, location and mass effect.",
            "Contrast-enhanced MRI to confirm the dural attachment.",
            "Small asymptomatic lesions are often managed with interval "
            "imaging (e.g. repeat MRI in 3–6 months); larger or symptomatic "
            "lesions may need surgery or radiotherapy.",
        ]
    if prediction == "Pituitary":
        return [
            "Endocrinology referral with a full pituitary hormone panel "
            "(prolactin, IGF-1, cortisol, TSH/T4, LH/FSH).",
            "Dedicated thin-slice pituitary MRI with contrast.",
            "Formal visual field assessment by ophthalmology.",
            "Neurosurgical review if the lesion is large or compresses the "
            "optic chiasm.",
        ]
    if prediction == "No Tumor":
        return [
            "Routine clinical follow-up as indicated by symptoms.",
            "If symptoms persist or progress, consider repeat or "
            "contrast-enhanced imaging.",
        ]
    return [
        "Full review of the images by a qualified radiologist.",
        "Repeat automated analysis once a validated model is available.",
    ]


def _precautions(prediction):
    if prediction == "Glioma":
        return [
            "Do not drive until cleared by the treating specialist, "
            "especially if any seizure has occurred.",
            "Avoid swimming or bathing alone, and working at heights, until "
            "seizure risk is assessed.",
            "Take any prescribed steroids or anti-seizure medication exactly "
            "as directed; do not stop abruptly.",
            RED_FLAGS,
            GENERAL_PRECAUTION,
        ]
    if prediction == "Meningioma":
        return [
            "Report any new headache pattern, vision change, weakness, "
            "personality change or seizure promptly.",
            "Keep scheduled follow-up imaging even if symptom-free.",
            RED_FLAGS,
            GENERAL_PRECAUTION,
        ]
    if prediction == "Pituitary":
        return [
            "Report any change in vision (blurring, loss of side vision, "
            "double vision) immediately.",
            "Seek urgent care for a sudden severe headache with vomiting or "
            "vision loss, which can indicate pituitary apoplexy.",
            "Watch for symptoms of hormone imbalance: fatigue, dizziness, "
            "weight change, menstrual or sexual changes, nipple discharge.",
            "If prescribed hormone replacement, do not miss doses; ask about "
            "sick-day steroid rules.",
            GENERAL_PRECAUTION,
        ]
    if prediction == "No Tumor":
        return [
            "Seek medical review if headaches worsen, or new neurological "
            "symptoms appear.",
            RED_FLAGS,
            GENERAL_PRECAUTION,
        ]
    return [
        "Do not treat this study as reassurance; no classification was "
        "performed.",
        GENERAL_PRECAUTION,
    ]


def clinical_notes(prediction, values):
    """Return {"impression", "recommendations", "precautions"}.

    prediction: "Glioma", "Meningioma", "Pituitary", "No Tumor" or empty
    values:     mapping with optional confidence / location / area / severity
    """
    prediction = (prediction or "").strip()
    values = values or {}
    return {
        "impression": _impression(prediction, values),
        "recommendations": _recommendations(prediction),
        "precautions": _precautions(prediction),
    }
