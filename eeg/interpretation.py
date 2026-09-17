"""Doctor-style wording for the EEG report.

Turns the verdict and the measured numbers into the prose a neurophysiology
report carries: a clinical impression, recommendations and precautions. The
same text feeds the web page and the PDF, so the two cannot disagree.

The precautions are standard patient-safety guidance for the finding, not a
treatment plan. Every item is framed for the reviewing clinician to confirm,
because the underlying call is a screening model's, not a diagnosis.
"""

TASK_NAMES = {
    "seizure": "epileptiform / seizure activity",
    "alzheimer": "cortical slowing (dementia screen)",
}


def _num(r, key, fmt, default="an unmeasured"):
    value = r.get(key)
    return fmt.format(value) if isinstance(value, (int, float)) else default


def _dominant_band(r):
    bands = r.get("band_means") or {}
    if not bands:
        return None
    return max(bands, key=bands.get)


def _impression(task, label, r):
    focus = TASK_NAMES.get(task, "the selected finding")
    duration = _num(r, "duration_s", "{:.0f} s of", "the submitted")
    band = _dominant_band(r)
    background = (f" The background is dominated by {band} activity."
                  if band else "")

    if label == "NOT ASSESSED":
        return (f"Screening for {focus} could not be performed: no validated "
                "model is installed for this question. Signal measurements "
                "are reported below, but no classification was made and this "
                "study offers no reassurance. Manual review of the full "
                "recording by a neurophysiologist is required." + background)

    if label == "NORMAL":
        return (f"{duration.capitalize()} recording was screened for {focus}. "
                "No episode crossed the detection threshold." + background +
                " A normal screening EEG does not exclude the condition: "
                "interictal recordings are frequently normal, and a single "
                "routine study has limited sensitivity.")

    episodes = r.get("episodes_n") or len(r.get("episodes") or []) or "One or more"
    first = (r.get("episodes") or [{}])[0]
    onset = _num(first, "onset_s", "{:.1f} s", "an unrecorded time")
    length = _num(first, "duration_s", "{:.1f} s", "an unrecorded duration")
    burden = _num(r, "burden_pct", "{:.1f}%", "an unmeasured share")

    if task == "alzheimer":
        return (f"ABNORMAL screening study. {duration.capitalize()} recording "
                f"shows {episodes} segment(s) consistent with cortical "
                f"slowing, the earliest at {onset} lasting {length}, "
                f"covering {burden} of the recording." + background +
                " Diffuse slowing is non-specific and can accompany "
                "neurodegenerative, metabolic, toxic or medication effects; "
                "clinical correlation is required.")
    return (f"ABNORMAL screening study. {duration.capitalize()} recording "
            f"shows {episodes} episode(s) of possible epileptiform activity, "
            f"the earliest at {onset} lasting {length}, covering {burden} of "
            "the recording." + background + " Findings are suggestive of an "
            "increased predisposition to seizures and require confirmation "
            "by a neurologist on the full multichannel recording.")


def _recommendations(task, label):
    if label == "NOT ASSESSED":
        return [
            "Full visual review of the raw recording by a qualified "
            "neurophysiologist.",
            "Repeat automated screening once a validated model is available.",
            "Correlate with the clinical history before any decision.",
        ]
    if label == "NORMAL":
        advice = [
            "Correlate with the clinical presentation; a normal study does "
            "not rule out the condition.",
            "If clinical suspicion persists, consider a repeat or prolonged "
            "EEG (sleep-deprived or ambulatory).",
        ]
        if task == "alzheimer":
            advice.append("Continue routine cognitive monitoring as "
                          "clinically indicated.")
        return advice
    if task == "alzheimer":
        return [
            "Neurology referral for formal cognitive assessment "
            "(e.g. MoCA / MMSE).",
            "Screen reversible causes of slowing: thyroid function, B12, "
            "electrolytes, glucose, renal and liver function, medication "
            "review.",
            "Consider structural neuroimaging (MRI brain) as clinically "
            "indicated.",
            "Repeat EEG to assess progression.",
        ]
    return [
        "Urgent neurology review to confirm the findings on the full "
        "multichannel recording.",
        "Consider prolonged video-EEG monitoring to characterise the "
        "events.",
        "MRI brain (epilepsy protocol) as clinically indicated.",
        "Review current medication, including anti-seizure drugs and any "
        "agents that lower seizure threshold.",
    ]


def _precautions(task, label):
    general = [
        "This report is a screening aid and must be verified by a qualified "
        "clinician before any clinical decision.",
    ]
    if label == "ABNORMAL" and task == "seizure":
        return [
            "Do not drive or operate heavy machinery until cleared by the "
            "treating neurologist, in line with local regulations.",
            "Avoid swimming or bathing alone, working at heights, and "
            "cooking over open flames without supervision.",
            "Maintain regular sleep; avoid sleep deprivation, alcohol and "
            "recreational drugs, which lower the seizure threshold.",
            "Take any prescribed anti-seizure medication exactly as directed; "
            "do not stop it abruptly.",
            "Carers: during a seizure, place the person on their side, keep "
            "them away from hazards and put nothing in the mouth.",
            "Call emergency services if a seizure lasts more than 5 minutes, "
            "seizures repeat without recovery, or there is injury or "
            "breathing difficulty.",
        ] + general
    if label == "ABNORMAL" and task == "alzheimer":
        return [
            "Arrange supervision for medication, finances and other complex "
            "tasks where memory or judgement may be affected.",
            "Review driving safety with the treating clinician.",
            "Make the home safe: reduce fall hazards, secure the stove, and "
            "consider an identification bracelet in case of wandering.",
            "Keep a regular routine, physical activity and social contact; "
            "manage blood pressure, diabetes and hearing loss.",
            "Seek prompt review for sudden confusion, which may indicate an "
            "acute medical cause rather than progression.",
        ] + general
    if label == "NOT ASSESSED":
        return [
            "Do not treat this study as reassurance; no classification was "
            "performed.",
        ] + general
    return [
        "Seek medical review if new symptoms occur: seizures, blackouts, "
        "unexplained confusion or memory decline.",
    ] + general


def clinical_notes(task, label, r):
    """Return {"impression", "recommendations", "precautions"} for a study.

    task:  "seizure" or "alzheimer"
    label: "ABNORMAL", "NORMAL" or "NOT ASSESSED"
    r:     the report payload (numbers may be missing; wording degrades,
           never raises)
    """
    r = r or {}
    return {
        "impression": _impression(task, label, r),
        "recommendations": _recommendations(task, label),
        "precautions": _precautions(task, label),
    }
