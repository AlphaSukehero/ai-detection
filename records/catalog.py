"""What each stored measurement means: label, unit, adult reference range.

Used by comparison and trend views. A range of None on either side means
there is no bound on that side; a key with no range is shown without a
normal/abnormal note. Ranges are general adult references, stated so the
clinician can see what a flag is based on -- not a diagnosis.
"""

CATALOG = {
    "ecg": [
        ("heart_rate", "Heart rate", "bpm", (60, 100)),
        ("rr_interval", "RR interval", "ms", (600, 1000)),
        ("pr_interval", "PR interval", "ms", (120, 200)),
        ("qrs_duration", "QRS duration", "ms", (None, 120)),
        ("qt_interval", "QT interval", "ms", None),
        ("qtc", "QTc", "ms", (None, 460)),
        ("axis", "QRS axis", "°", (-30, 90)),
        ("sdnn", "SDNN", "ms", None),
        ("rmssd", "RMSSD", "ms", None),
        ("classifier_confidence", "Classifier confidence", "%", None),
    ],
    "eeg": [
        ("burden_pct", "Anomaly burden", "%", None),
        ("episodes", "Anomalous episodes", "", None),
        ("mean_spike_rate", "Mean spike rate", "/s", None),
        ("rsp_delta", "Relative delta power", "", None),
        ("rsp_theta", "Relative theta power", "", None),
        ("rsp_alpha", "Relative alpha power", "", None),
        ("rsp_beta", "Relative beta power", "", None),
        ("rsp_gamma", "Relative gamma power", "", None),
        ("duration_s", "Recording duration", "s", None),
    ],
    "mri": [
        ("mri_confidence", "Classifier confidence", "%", None),
        ("p_glioma", "P(glioma)", "%", None),
        ("p_meningioma", "P(meningioma)", "%", None),
        ("p_notumor", "P(no tumour)", "%", None),
        ("p_pituitary", "P(pituitary)", "%", None),
    ],
}

MODALITY_LABELS = {"ecg": "ECG", "eeg": "EEG", "mri": "Brain MRI"}


def entries(modality):
    return CATALOG.get(modality, [])


def lookup(modality, key):
    for k, label, unit, rng in entries(modality):
        if k == key:
            return label, unit, rng
    return key, "", None


def range_status(value, rng):
    """'below normal' / 'normal' / 'above normal', or None if unknowable."""
    if value is None or rng is None:
        return None
    lo, hi = rng
    if lo is not None and value < lo:
        return "below normal"
    if hi is not None and value > hi:
        return "above normal"
    return "normal"
