"""Beat segmentation and RR-interval context, shared by training and inference.

This module exists so the preparation script and the Flask app derive their
model inputs from one implementation. Any divergence between how beats are
cut at training time and at serving time is a silent accuracy loss that no
test of either side alone would catch.

Why RR features at all: the beat window is z-scored and centred on the R
peak, which normalises away exactly the information that defines a
supraventricular ectopic beat. An S beat is not primarily an odd shape, it is
an *early* beat -- its prematurity lives in the interval before it, not in
its morphology. A classifier given only the window cannot see that, which is
why S recall sits near zero on an inter-patient split however long it trains.

The four features are ratios against the record's own median RR rather than
absolute seconds. That is deliberate and is what makes them transfer between
patients: a 0.7 s interval is bradycardic in one record and tachycardic in
another, but "0.6x this patient's typical interval" means the same thing in
both.
"""
import numpy as np

BEFORE, AFTER = 100, 180          # 280 samples at 360 Hz, R peak at index 100
BEAT_LEN = BEFORE + AFTER
LOCAL_WINDOW = 10                 # beats either side for the local RR average
N_RR_FEATURES = 4


def rr_features(peaks, fs=360.0):
    """Per-beat RR context, shape (len(peaks), 4).

    Columns: pre_RR, post_RR and local_RR as ratios of the record's median
    RR, plus the pre/post ratio. Edge beats reuse the nearest available
    interval rather than being dropped, so the array aligns 1:1 with `peaks`
    and callers never have to reconcile two different lengths.
    """
    peaks = np.asarray(peaks, dtype=float)
    n = len(peaks)
    if n == 0:
        return np.zeros((0, N_RR_FEATURES), dtype=np.float32)
    if n == 1:
        # One beat carries no interval information. Neutral ratios say
        # "typical" rather than fabricating a rhythm from a single point.
        return np.ones((1, N_RR_FEATURES), dtype=np.float32)

    rr = np.diff(peaks) / fs                      # len n-1
    pre = np.concatenate([[rr[0]], rr])           # interval before each beat
    post = np.concatenate([rr, [rr[-1]]])         # interval after each beat

    median_rr = float(np.median(rr))
    if median_rr <= 0:
        return np.ones((n, N_RR_FEATURES), dtype=np.float32)

    local = np.empty(n, dtype=float)
    for i in range(n):
        lo = max(0, i - LOCAL_WINDOW)
        hi = min(len(rr), i + LOCAL_WINDOW)
        window = rr[lo:hi] if hi > lo else rr
        local[i] = float(np.mean(window))

    feats = np.stack([pre / median_rr,
                      post / median_rr,
                      local / median_rr,
                      pre / np.where(post > 0, post, median_rr)], axis=1)
    # Ratios beyond this are artefacts of missed or spurious peak detections,
    # not physiology; clipping stops one bad interval dominating the input.
    return np.clip(feats, 0.0, 4.0).astype(np.float32)


def extract_beats(signal, peaks, fs=360.0):
    """Z-scored windows centred on each peak, with their RR context.

    Returns (beats, features, kept) where `kept` indexes the peaks that had a
    full window inside the signal. RR context is computed over ALL peaks
    before dropping edge beats, so a discarded first beat still contributes
    its interval to its neighbour.
    """
    signal = np.asarray(signal, dtype=np.float32).ravel()
    peaks = np.asarray(peaks, dtype=int)
    feats_all = rr_features(peaks, fs)

    beats, kept = [], []
    for i, peak in enumerate(peaks):
        start, end = peak - BEFORE, peak + AFTER
        if start < 0 or end > len(signal):
            continue
        beat = signal[start:end]
        std = float(beat.std())
        beats.append((beat - beat.mean()) / (std if std > 1e-6 else 1.0))
        kept.append(i)

    if not beats:
        return (np.zeros((0, BEAT_LEN), np.float32),
                np.zeros((0, N_RR_FEATURES), np.float32),
                np.zeros(0, dtype=int))
    kept = np.asarray(kept, dtype=int)
    return np.asarray(beats, np.float32), feats_all[kept], kept


def segment_signal(signal, fs=360.0):
    """Detect R peaks and return (beats, rr_features) for inference.

    The counterpart of what scripts/prepare_ecg.py does with the reference
    annotations. Imported lazily so this module stays importable without the
    delineation stack.
    """
    from ecg.delineate import detect_r_peaks
    peaks = detect_r_peaks(signal, fs)
    beats, feats, _ = extract_beats(signal, peaks, fs)
    return beats, feats
