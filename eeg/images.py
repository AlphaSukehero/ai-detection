"""Turn a signal window into the time-frequency image the model is shown.

Two representations, both computed from the signal rather than from a
screenshot of a plot: a Short-Time Fourier spectrogram and a Continuous
Wavelet Transform scalogram. The scalogram is the default because Morlet
wavelets give better time localisation at low frequencies, which is where the
clinical content lives -- delta slowing and spike-wave discharges both sit
below 13 Hz, and an STFT window long enough to resolve 2 Hz smears a 200 ms
spike across most of it.

The arrays produced here are model input, never a rendering of a rendering.
Nothing in this module reads pixels.
"""
import numpy as np
import pywt
from scipy.signal import spectrogram as _stft_spectrogram

IMAGE_SIZE = (64, 64)              # (frequency bins, time bins)
FREQ_RANGE = (0.5, 45.0)           # matches eeg.metrics.ANALYSIS_BAND
WAVELET = "morl"


def _resize(arr, shape):
    """Bilinear resample without pulling in a heavyweight image dependency."""
    src_r, src_c = arr.shape
    dst_r, dst_c = shape
    if (src_r, src_c) == (dst_r, dst_c):
        return arr
    rows = np.linspace(0, src_r - 1, dst_r)
    cols = np.linspace(0, src_c - 1, dst_c)
    r0 = np.clip(np.floor(rows).astype(int), 0, src_r - 1)
    r1 = np.clip(r0 + 1, 0, src_r - 1)
    c0 = np.clip(np.floor(cols).astype(int), 0, src_c - 1)
    c1 = np.clip(c0 + 1, 0, src_c - 1)
    wr = (rows - r0)[:, None]
    wc = (cols - c0)[None, :]
    top = arr[np.ix_(r0, c0)] * (1 - wc) + arr[np.ix_(r0, c1)] * wc
    bot = arr[np.ix_(r1, c0)] * (1 - wc) + arr[np.ix_(r1, c1)] * wc
    return top * (1 - wr) + bot * wr


def _normalise(power):
    """Log-compress, then scale to [0, 1] per image.

    Per-image rather than per-dataset: EEG amplitude varies with electrode
    impedance and montage, and a global scale would let the model separate
    recordings by gain instead of by physiology. The trade-off is that
    absolute power is no longer recoverable from the image -- which is fine,
    because absolute power is reported from eeg.metrics, computed on the
    signal, not read back off a picture.
    """
    # The floor is relative to this image's peak. A fixed absolute floor
    # breaks gain invariance: scaling the signal moves the peak but not the
    # clipped minimum, so the same rhythm at two gains normalised differently.
    peak = float(np.max(power)) if power.size else 0.0
    with np.errstate(divide="ignore"):
        logp = np.log10(np.maximum(power, peak * 1e-20))
    lo, hi = float(logp.min()), float(logp.max())
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo < 1e-12:
        return np.zeros_like(logp)
    return (logp - lo) / (hi - lo)


def scalogram(x, fs, size=IMAGE_SIZE, freq_range=FREQ_RANGE):
    """CWT magnitude on a log-spaced frequency grid, normalised to [0, 1]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n_freq, n_time = size
    if x.size < 8:
        return np.zeros(size, dtype=np.float32)

    freqs = np.logspace(np.log10(freq_range[0]), np.log10(freq_range[1]), n_freq)
    # pywt scales relate to frequency through the wavelet's centre frequency.
    scales = pywt.central_frequency(WAVELET) * fs / freqs
    coeffs, _ = pywt.cwt(x, scales, WAVELET, sampling_period=1.0 / fs)
    power = np.abs(coeffs) ** 2
    return _resize(_normalise(power), size).astype(np.float32)


def spectrogram(x, fs, size=IMAGE_SIZE, freq_range=FREQ_RANGE):
    """STFT power spectrogram, normalised to [0, 1]."""
    x = np.asarray(x, dtype=np.float64).ravel()
    n_freq, n_time = size
    if x.size < 8:
        return np.zeros(size, dtype=np.float32)

    nperseg = max(8, min(len(x) // 4, int(round(fs * 0.5))))
    freqs, _, power = _stft_spectrogram(
        x, fs=fs, nperseg=nperseg, noverlap=nperseg // 2)
    keep = (freqs >= freq_range[0]) & (freqs <= freq_range[1])
    power = power[keep]
    if power.size == 0 or power.shape[1] == 0:
        return np.zeros(size, dtype=np.float32)
    return _resize(_normalise(power), size).astype(np.float32)


def render(x, fs, kind="scalogram", size=IMAGE_SIZE):
    if kind == "scalogram":
        return scalogram(x, fs, size)
    if kind == "spectrogram":
        return spectrogram(x, fs, size)
    raise ValueError(f"unknown representation {kind!r}")


def render_batch(windows, fs, kind="scalogram", size=IMAGE_SIZE):
    """Render many windows to (n, freq, time, 1) ready for the model.

    A multi-channel window is averaged across channels first. That is a real
    simplification -- it discards spatial localisation, so this pipeline can
    say a discharge occurred but not which electrodes led it. Recorded here
    because a reader should not have to infer it from the shape.
    """
    out = []
    for w in np.asarray(windows):
        sig = np.mean(w, axis=0) if np.ndim(w) == 2 else w
        out.append(render(sig, fs, kind, size))
    if not out:
        return np.zeros((0,) + size + (1,), dtype=np.float32)
    return np.stack(out)[..., np.newaxis]
