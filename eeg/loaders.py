"""Read recordings and their clinical annotations into signal + label form.

MNE is imported lazily and behind a flag, matching how this repository treats
pydicom and OpenCV: the signal maths, the windowing and the metrics are all
testable without it, and only the file readers actually need it.

Two corpora, two label shapes, and the difference matters:

  * CHB-MIT gives per-event seizure onset/offset in seconds, so windows can
    be labelled individually and the model learns to localise in time.
  * ds004504 gives one diagnosis per subject, with no indication of which
    moments carry the biomarker. Propagating that label to every window is a
    known weak-labelling compromise: a window inherits "Alzheimer's" from its
    subject, not from anything measured in the window. Said plainly here
    because it caps what the dementia head can honestly claim.
"""
import os
import re

try:
    import mne
    mne.set_log_level("ERROR")
    MNE_AVAILABLE = True
except Exception as e:                      # pragma: no cover - env dependent
    print("MNE import warning:", e)
    MNE_AVAILABLE = False


class RecordingError(RuntimeError):
    """A recording could not be read, or carries no usable timebase."""


def _require_mne():
    if not MNE_AVAILABLE:
        raise RecordingError(
            "MNE is not installed, so recordings cannot be read. "
            "pip install -r requirements-eeg.txt"
        )


def read_recording(path, picks=None, l_freq=0.5, h_freq=45.0):
    """Load an EDF/BDF/SET file. Returns (signal, fs, channel_names).

    Band-pass filtered to the analysis range on load so every consumer sees
    the same conditioning. The filter is applied here rather than per window
    because filtering short windows independently introduces edge transients
    that look like spikes.
    """
    _require_mne()
    if not os.path.exists(path):
        raise RecordingError(f"no such recording: {path}")

    ext = os.path.splitext(path)[1].lower()
    try:
        if ext in (".edf", ".bdf"):
            raw = mne.io.read_raw_edf(path, preload=True, verbose="ERROR")
        elif ext == ".set":
            raw = mne.io.read_raw_eeglab(path, preload=True, verbose="ERROR")
        elif ext == ".fif":
            raw = mne.io.read_raw_fif(path, preload=True, verbose="ERROR")
        else:
            raise RecordingError(f"unsupported recording format {ext!r}")
    except RecordingError:
        raise
    except Exception as e:
        raise RecordingError(f"could not read {os.path.basename(path)}: {e}") from e

    if picks:
        available = [c for c in picks if c in raw.ch_names]
        if available:
            raw.pick(available)

    fs = float(raw.info["sfreq"])
    if fs <= 0:
        raise RecordingError("recording declares no sampling rate")
    # Nyquist guard: asking for 45 Hz from a 64 Hz recording is not a filter
    # error to swallow, but the cutoff must be lowered rather than assumed.
    top = min(h_freq, fs / 2.0 - 1.0)
    if top > l_freq:
        raw.filter(l_freq, top, verbose="ERROR")

    # MNE returns SI units -- volts -- so a normal 20 uV rhythm arrives as
    # 2e-5. Every threshold downstream (artifact rejection, spike amplitude)
    # is expressed in microvolts because that is how EEG is read clinically,
    # so the conversion happens once, here, rather than being re-derived at
    # each call site. Without it a healthy recording reads as flatline and
    # every window is rejected.
    return raw.get_data() * 1e6, fs, list(raw.ch_names)


# --------------------------------------------------------------- CHB-MIT

_SEIZURE_START = re.compile(r"Seizure.*Start Time:\s*(\d+)\s*seconds")
_SEIZURE_END = re.compile(r"Seizure.*End Time:\s*(\d+)\s*seconds")


def parse_chb_summary(summary_path):
    """Map each EDF filename to its list of (onset_s, offset_s) seizures.

    Files with no seizures map to an empty list and are kept in the mapping:
    they are the negative class, and dropping them here is the same mistake
    as dropping baseline windows later.
    """
    if not os.path.exists(summary_path):
        raise RecordingError(f"no CHB-MIT summary at {summary_path}")

    seizures, current, starts = {}, None, []
    with open(summary_path) as f:
        for line in f:
            line = line.strip()
            if line.startswith("File Name:"):
                current = line.split(":", 1)[1].strip()
                seizures.setdefault(current, [])
                starts = []
            elif current:
                m = _SEIZURE_START.search(line)
                if m:
                    starts.append(int(m.group(1)))
                    continue
                m = _SEIZURE_END.search(line)
                if m and starts:
                    seizures[current].append((float(starts.pop(0)),
                                              float(m.group(1))))
    return seizures


def load_chb_record(edf_path, summary):
    """Return (signal, fs, annotations) for one CHB-MIT recording."""
    signal, fs, _ = read_recording(edf_path)
    return signal, fs, summary.get(os.path.basename(edf_path), [])


# -------------------------------------------------------------- ds004504

def parse_participants(tsv_path):
    """Map subject id -> diagnosis group (A, C or F)."""
    if not os.path.exists(tsv_path):
        raise RecordingError(f"no participants file at {tsv_path}")
    groups = {}
    with open(tsv_path) as f:
        header = f.readline().rstrip("\n").split("\t")
        try:
            sid, gid = header.index("participant_id"), header.index("Group")
        except ValueError as e:
            raise RecordingError(
                f"participants.tsv lacks participant_id/Group: {header}") from e
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) > max(sid, gid):
                groups[parts[sid]] = parts[gid]
    return groups
