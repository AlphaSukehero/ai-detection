"""EEG Image Analysis dashboard.

    .venv/bin/streamlit run eeg_app.py

Three tabs; the middle one enforces the sequential workflow:

    Step 1  Patient intake -- nothing else unlocks until it is complete
    Step 2  Upload (EDF/BDF/SET, or a plot image plus its duration)
    Step 3  Analysis: timeline, episodes, per-window clinical parameters

Two things this dashboard will not do, both deliberate:

  * It will not report a frequency-domain parameter from an image with no
    timebase. Hertz needs seconds; a picture has pixels. It asks for the
    duration instead, and refuses if it cannot get one.
  * It will not describe a recording as normal when no model was available.
    "Not assessed" and "nothing found" are different statements and the
    difference matters more here than anywhere else in the page.
"""
import os

import numpy as np
import streamlit as st

from eeg.digitize import CalibrationError, digitize
from eeg.inference import analyse_signal, summarise
from eeg.loaders import RecordingError, read_recording
from eeg.metrics import BANDS

st.set_page_config(page_title="EEG Analysis", layout="wide")

MODEL_DIR = "model"
TASKS = {"seizure": "Epileptiform / seizure activity",
         "alzheimer": "Cortical slowing (dementia screen)"}
SIGNAL_EXT = {".edf", ".bdf", ".set", ".fif"}
IMAGE_EXT = {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".bmp"}
REQUIRED = ["patient_id", "age", "gender", "history"]


# ----------------------------------------------------------------- model

@st.cache_resource(show_spinner=False)
def load_task_model(task):
    """Return (model, card) or (None, None). A missing model is normal here:
    the measured parameters do not depend on one."""
    path = os.path.join(MODEL_DIR, f"eeg_{task}.keras")
    try:
        from mlkit.registry import load_card
        import tensorflow as tf
        card = load_card(path)
        return tf.keras.models.load_model(path, compile=False), card
    except Exception as e:
        print(f"EEG model {task} unavailable: {e}")
        return None, None


# ------------------------------------------------------------ step one

def patient_form():
    st.subheader("Step 1 — Patient information")
    st.caption("Required before any recording can be uploaded.")
    saved = st.session_state.get("patient", {})

    with st.form("patient_form"):
        c1, c2 = st.columns(2)
        with c1:
            pid = st.text_input("Patient ID / MRN *", saved.get("patient_id", ""))
            age = st.number_input("Age *", 0, 120,
                                  int(saved.get("age", 0)), step=1)
        with c2:
            gender = st.selectbox(
                "Gender *", ["", "Male", "Female", "Other", "Prefer not to say"],
                index=0)
            referrer = st.text_input("Referring clinician",
                                     saved.get("referrer", ""))
        history = st.text_area(
            "Clinical history / indication *", saved.get("history", ""),
            placeholder="Seizure semiology, medication, prior imaging, "
                        "cognitive complaints...")
        submitted = st.form_submit_button("Save and continue")

    if submitted:
        missing = []
        if not pid.strip():
            missing.append("Patient ID")
        if age <= 0:
            missing.append("Age")
        if not gender:
            missing.append("Gender")
        if not history.strip():
            missing.append("Clinical history")
        if missing:
            st.error("Required: " + ", ".join(missing))
            return False
        st.session_state["patient"] = {
            "patient_id": pid.strip(), "age": int(age), "gender": gender,
            "referrer": referrer.strip() or "Not provided",
            "history": history.strip(),
        }
        st.success("Saved. Upload unlocked below.")
        return True
    return bool(st.session_state.get("patient"))


def patient_complete():
    p = st.session_state.get("patient") or {}
    return all(p.get(k) for k in REQUIRED)


# ------------------------------------------------------------ step two

def upload_section():
    st.subheader("Step 2 — Upload recording")
    task = st.selectbox("Clinical question", list(TASKS),
                        format_func=lambda k: TASKS[k])
    uploaded = st.file_uploader(
        "EEG recording", type=[e[1:] for e in sorted(SIGNAL_EXT | IMAGE_EXT)],
        help="EDF/BDF/SET is preferred. An image of a plotted trace also "
             "works, but needs its recording duration to establish a "
             "timebase.")
    if uploaded is None:
        return None

    ext = os.path.splitext(uploaded.name)[1].lower()
    duration = None
    if ext in IMAGE_EXT:
        st.info(
            "This is an image, so it carries pixels rather than samples. "
            "Relative spectral power, spectral entropy and spike rate are all "
            "defined in hertz, so the recording duration is needed to recover "
            "a timebase. Without it the analysis is refused rather than "
            "estimated.")
        duration = st.number_input(
            "Total duration of this recording (seconds)",
            0.0, 86400.0, 0.0, step=1.0,
            help="Printed on most clinical exports. Leave at 0 to rely on a "
                 "detected calibration grid, if the image has one.")
        duration = duration or None
    return {"file": uploaded, "ext": ext, "task": task, "duration": duration}


def to_signal(upload, target_fs=128.0):
    """Upload -> (signal, fs, provenance). Raises on an unusable input."""
    data, ext = upload["file"], upload["ext"]
    tmp = os.path.join("/tmp", f"eeg_upload{ext}")
    with open(tmp, "wb") as f:
        f.write(data.getbuffer())

    if ext in SIGNAL_EXT:
        signal, fs, names = read_recording(tmp)
        # Average across channels: this pipeline localises in time, not in
        # space. It can say when a discharge occurred, not which electrodes
        # led it.
        return np.mean(signal, axis=0), fs, f"{len(names)} channels at {fs:.0f} Hz"

    signal, fs, source = digitize(tmp, duration_s=upload["duration"],
                                  target_fs=target_fs)
    return signal, fs, f"digitised from image ({source})"


# ---------------------------------------------------------- step three

def _band_table(result):
    rows = []
    for w in result["windows"]:
        if w["rsp"] is None:
            continue
        rows.append({"t (s)": round(w["start_s"], 1),
                     **{b: round(w["rsp"][b], 3) for b in BANDS},
                     "H_spec": (round(w["spectral_entropy"], 3)
                                if w["spectral_entropy"] is not None else None),
                     "spikes/s": round(w["spikes"]["rate_per_s"], 2),
                     "score": (round(w["score"], 3)
                               if w.get("score") is not None else None),
                     "artifact": w["artifact"] or ""})
    return rows


def analysis_panel(upload):
    st.subheader("Step 3 — Analysis")
    try:
        signal, fs, provenance = to_signal(upload)
    except CalibrationError as e:
        st.error(str(e))
        return
    except RecordingError as e:
        st.error(f"Could not read this recording: {e}")
        return

    st.caption(f"Source: {provenance} — {len(signal) / fs:.1f} s")
    model, card = load_task_model(upload["task"])
    if model is None:
        st.warning(
            "No validated model is installed for this task, so no window will "
            "be classified. Every measurement below is computed from the "
            "signal itself and remains valid. This is **not** a statement "
            "that the recording is normal.")

    result = analyse_signal(signal, fs, model=model, card=card)
    if result["n_windows"] == 0:
        st.error(result.get("error", "Nothing to analyse."))
        return
    summary = summarise(result)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Windows", result["n_windows"])
    c2.metric("Episodes", summary["episodes"] if result["model_used"] else "—")
    c3.metric("Anomaly burden",
              f"{summary['burden_pct']:.1f}%" if result["model_used"] else "—")
    c4.metric("Artifact windows", result.get("artifact_windows", 0))
    st.info(summary["headline"])

    st.markdown("#### Timeline")
    if result["model_used"]:
        st.line_chart({"anomaly score": [w["score"] for w in result["windows"]]},
                      x_label="window", y_label="score")
    st.line_chart(
        {b: [(w["rsp"][b] if w["rsp"] else None) for w in result["windows"]]
         for b in BANDS},
        x_label="window", y_label="relative spectral power")

    if result["episodes"]:
        st.markdown("#### Detected episodes")
        st.dataframe([{
            "onset (s)": round(e["onset_s"], 1),
            "offset (s)": round(e["offset_s"], 1),
            "duration (s)": round(e["duration_s"], 1),
            "peak score": round(e["peak_score"], 3),
            "spikes/s": round(e["spike_rate_per_s"], 2),
        } for e in result["episodes"]], width="stretch")
    elif result["model_used"]:
        st.success("No episode crossed the detection threshold.")

    with st.expander("Per-window clinical parameters"):
        st.caption(
            "RSP shares sum to 1 across the analysis band. H_spec is Shannon "
            "entropy of the normalised power spectrum, 0 = pure rhythm, "
            "1 = broadband. Blank rows were unmeasurable.")
        st.dataframe(_band_table(result), width="stretch")

    if card:
        with st.expander("Model card — what this model was measured at"):
            st.json(card)

    st.caption(
        "Research prototype, not a medical device. Every value above is "
        "model- or signal-derived and must be reviewed by a qualified "
        "clinician before any decision.")
    st.session_state["last_result"] = result


# ------------------------------------------------------------------ main

tab_analysis, tab_metrics, tab_about = st.tabs(
    ["EEG Image Analysis", "Parameters & formulas", "About"])

with tab_analysis:
    st.title("EEG Image Analysis")
    done = patient_form()
    st.divider()
    if not patient_complete():
        st.warning("Complete the patient intake form above to unlock upload.")
    else:
        p = st.session_state["patient"]
        st.caption(f"Patient {p['patient_id']} · {p['age']} · {p['gender']}")
        upload = upload_section()
        st.divider()
        if upload:
            analysis_panel(upload)
        else:
            st.info("Upload a recording to run the analysis.")

with tab_metrics:
    st.title("Parameters and formulas")
    st.markdown(r"""
**Relative spectral power** — share of band power within the analysis band:

$$\mathrm{RSP}_{[f_1,f_2]}=\frac{\int_{f_1}^{f_2}P(f)\,df}{\int_{0.5}^{45}P(f)\,df}$$

with $P(f)$ the Welch power spectral density. Bands tile 0.5–45 Hz, so the
reported shares sum to 1.

**Spectral entropy** — Shannon entropy of the normalised spectrum:

$$p_k=\frac{P(f_k)}{\sum_j P(f_j)},\qquad
H_{\mathrm{spec}}=-\sum_k p_k\log_2 p_k,\qquad
H_{\mathrm{norm}}=\frac{H_{\mathrm{spec}}}{\log_2 N}$$

0 is a pure rhythm, 1 is broadband. Normalising by $\log_2 N$ keeps windows of
different length comparable.

**Spike density** — paroxysmal discharges per second:

$$\sigma_{\mathrm{MAD}}=1.4826\,\mathrm{median}(|x-\mathrm{median}(x)|),\qquad
\text{rate}=\frac{|\{\text{peaks}:|x-\mathrm{median}|>5\sigma_{\mathrm{MAD}}\}|}{T}$$

A median-based sigma is used because an ordinary standard deviation is
inflated by the very spikes being counted, so a dense burst would raise its
own threshold and hide.

**Onset and duration** — consecutive flagged windows merge into one episode
running from the start of the first to the end of the last:

$$t_{\mathrm{onset}}=\min_i t^{\mathrm{start}}_i,\quad
t_{\mathrm{offset}}=\max_i t^{\mathrm{stop}}_i,\quad
\Delta=t_{\mathrm{offset}}-t_{\mathrm{onset}}$$
""")

with tab_about:
    st.title("About")
    st.markdown("""
**Pipeline.** Recordings are band-passed to 0.5–45 Hz, split into 2-second
windows at 50% overlap, screened for artifact, and rendered as Morlet wavelet
scalograms. The same windowing code runs at training and inference, so the
two cannot drift apart.

**Uploaded images** are digitised back to a signal before anything else. The
slicing happens in signal space, not pixel space — cutting a plot into image
strips would feed the model a picture of a picture, at whatever resolution the
crop landed on. An image with no recoverable timebase is refused.

**Artifact handling.** EMG bursts, flatline, saturation and gross excursions
are detected and never flagged as clinical findings. Baseline is kept as the
negative class rather than discarded; a model trained only on anomalies has no
decision boundary and reports findings everywhere.

**Limitations.** Channels are averaged, so findings are localised in time but
not in space. The dementia head is trained on subject-level labels applied to
every window, which is weak labelling and caps what it can claim. Consult each
model's card for the split it was measured on.
""")
