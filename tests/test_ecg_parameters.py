import numpy as np
from ecg.parameters import heart_rate, rhythm
from ecg.quality import OK, UNAVAILABLE


def test_heart_rate_from_evenly_spaced_peaks():
    peaks = np.arange(0, 3600, 288)          # 0.8 s spacing at 360 Hz -> 75 bpm
    m = heart_rate(peaks, fs=360.0)
    assert m.quality == OK
    assert abs(m.value - 75.0) < 0.5


def test_heart_rate_unavailable_with_too_few_peaks():
    m = heart_rate(np.array([10, 300]), fs=360.0)
    assert m.quality == UNAVAILABLE
    assert m.value is None


def test_regular_rhythm_for_constant_rr():
    peaks = np.arange(0, 3600, 288)
    m = rhythm(peaks, fs=360.0)
    assert m.reason == "Regular"
    assert m.value < 0.10


def test_irregular_rhythm_for_varying_rr():
    peaks = np.array([0, 200, 620, 780, 1300, 1450, 2000, 2100])
    m = rhythm(peaks, fs=360.0)
    assert m.reason == "Irregular"


from ecg.parameters import pr_interval
from ecg.quality import UNAVAILABLE


def _beat_with_p_wave(fs=360.0, n_beats=6, rr=0.8):
    n = int(n_beats * rr * fs)
    sig = np.zeros(n)
    for b in range(n_beats):
        r = int(b * rr * fs) + 100
        if r + 20 >= n:
            break
        # Dense QRS ~70 ms wide: a real complex occupies every sample it spans,
        # and a width measured from isolated spikes is not physiological.
        half = int(0.035 * fs)
        for off in range(-half, half + 1):
            if 0 <= r + off < n:
                sig[r + off] = 3.0 * (1.0 - abs(off) / (half + 1.0))
        p = r - int(0.16 * fs)             # P wave 160 ms before R
        pw = int(0.02 * fs)
        if p - pw > 0:
            for off in range(-pw, pw + 1):
                sig[p + off] = 0.45 * (1.0 - abs(off) / (pw + 1.0))
    return sig


def test_pr_interval_measured_near_expected_value():
    sig = _beat_with_p_wave()
    m = pr_interval(sig, None, fs=360.0)
    assert m.value is not None
    assert 0.10 <= m.value <= 0.22


def test_pr_unavailable_when_no_p_wave():
    fs = 360.0
    sig = np.zeros(int(6 * 0.8 * fs))
    for b in range(6):
        r = int(b * 0.8 * fs) + 100
        if r + 2 < len(sig):
            sig[r] = 3.0
            sig[r - 1] = sig[r + 1] = 1.0
    m = pr_interval(sig, None, fs=fs)
    assert m.value is None
    assert m.quality == UNAVAILABLE


from ecg.parameters import qrs_duration, qt_interval, qtc_fridericia


def test_qtc_fridericia_matches_formula():
    assert abs(qtc_fridericia(0.40, 1.0) - 0.40) < 1e-9
    assert abs(qtc_fridericia(0.40, 0.512) - 0.50) < 1e-3


def test_qtc_fridericia_is_tamer_than_bazett_at_high_rate():
    """At 195 bpm Bazett produced 722 ms; Fridericia must stay far below."""
    rr = 0.307
    assert qtc_fridericia(0.40, rr) * 1000 < 620


def test_qrs_duration_in_physiological_range():
    sig = _beat_with_p_wave()
    m = qrs_duration(sig, None, fs=360.0)
    assert m.value is not None
    # The fixture builds a QRS 2*0.035 s wide; a bound of 0.02-0.20 s would
    # be satisfied by anything the plausibility filter already lets through,
    # so assert the fixture's known width instead.
    assert abs(m.value - 0.070) < 0.015, m.value


def _beat_with_t_wave(fs=360.0, n_beats=6, rr=0.8, t_amp=0.8):
    """_beat_with_p_wave plus a real T wave, so QT is genuinely measurable.

    The P-wave fixture has no T wave at all: any QT it yielded came from the
    tangent extrapolation being clamped to the search-window edge, which is
    the fabrication ecg.delineate.t_end now rejects.
    """
    sig = _beat_with_p_wave(fs=fs, n_beats=n_beats, rr=rr)
    half = int(0.06 * fs)                        # T wave 120 ms wide
    for b in range(n_beats):
        r = int(b * rr * fs) + 100
        centre = r + int(0.24 * fs)              # T apex ~240 ms after R
        for off in range(-half, half + 1):
            i = centre + off
            if 0 <= i < len(sig):
                sig[i] += t_amp * np.cos(0.5 * np.pi * off / half) ** 2
    return sig


def test_qt_returns_measurement_not_constant():
    sig = _beat_with_t_wave()
    m = qt_interval(sig, None, fs=360.0)
    # "None or != 0.40" would pass when QT is unmeasurable - exactly the
    # failure this test exists to catch. Require a real measurement.
    assert m.value is not None
    assert m.value != 0.40
    assert 0.20 <= m.value <= 0.65


from ecg.delineate import detect_r_peaks
from ecg.parameters import st_deviation


def test_st_normal_for_flat_baseline():
    sig = _beat_with_p_wave()
    m = st_deviation(sig, None, fs=360.0)
    assert m.value is not None
    assert m.reason == "Normal", f"{m.reason} ({m.value:+.2f} mm)"
    assert abs(m.value) <= 1.0


def test_st_elevation_detected_when_segment_raised():
    fs = 360.0
    sig = _beat_with_p_wave(fs=fs)
    peaks = detect_r_peaks(sig, fs)
    for peak in peaks:
        j = int(peak) + int(0.04 * fs)
        sig[j:j + int(0.10 * fs)] += 0.9      # lift the ST segment
    m = st_deviation(sig, peaks, fs=fs)
    assert m.reason == "Elevated"


def test_st_unavailable_without_peaks():
    m = st_deviation(np.zeros(1000), np.array([]), fs=360.0)
    assert m.value is None


from ecg.parameters import qrs_axis
from ecg.quality import UNAVAILABLE


def test_axis_unavailable_for_single_lead():
    m = qrs_axis({"II": _beat_with_p_wave()}, fs=360.0)
    assert m.value is None
    assert m.quality == UNAVAILABLE
    assert m.reason == "Requires 12-lead"


def test_axis_near_zero_when_lead_I_positive_and_aVF_flat():
    sig = _beat_with_p_wave()
    flat = np.zeros_like(sig)
    m = qrs_axis({"I": sig, "aVF": flat}, fs=360.0)
    assert m.value is not None
    assert abs(m.value) < 20.0


def test_axis_near_ninety_when_aVF_dominant():
    sig = _beat_with_p_wave()
    flat = np.zeros_like(sig)
    m = qrs_axis({"I": flat, "aVF": sig}, fs=360.0)
    assert m.value is not None
    assert abs(m.value - 90.0) < 20.0


from ecg.parameters import analyse


def test_analyse_returns_every_expected_parameter():
    report = analyse(_beat_with_p_wave(), fs=360.0)
    for key in ["heart_rate", "rhythm", "pr_interval", "qrs_duration",
                "qt_interval", "qtc", "st_segment", "axis"]:
        assert key in report, key


def test_analyse_never_emits_the_old_hardcoded_values():
    """Guards the two fabricated constants this work removes."""
    report = analyse(_beat_with_p_wave(), fs=360.0)
    pr = report["pr_interval"]
    qt = report["qt_interval"]
    assert not (pr.value is not None and abs(pr.value - 0.145) < 1e-9)
    assert not (qt.value is not None and abs(qt.value - 0.400) < 1e-9)


def test_analyse_marks_axis_unavailable_for_single_lead():
    report = analyse(_beat_with_p_wave(), fs=360.0)
    assert report["axis"].reason == "Requires 12-lead"


def test_analyse_accepts_a_dict_of_leads():
    sig = _beat_with_p_wave()
    report = analyse({"I": sig, "aVF": np.zeros_like(sig), "II": sig}, fs=360.0)
    assert report["heart_rate"].value is not None


def test_analyse_skips_blank_leads_when_choosing_primary():
    """A None-valued lead II must not crash or become the primary signal."""
    sig = _beat_with_p_wave()
    report = analyse({"II": None, "I": sig}, fs=360.0)
    assert report["heart_rate"].value is not None


def test_analyse_reports_all_unavailable_when_every_lead_is_blank():
    report = analyse({"I": None, "II": None}, fs=360.0)
    assert report["heart_rate"].value is None
    assert report["display"]["heart_rate"] == "—"


def test_everything_unavailable_from_image_without_grid_scale():
    """No grid means no timebase and no gain, so NOTHING is measurable.

    ST is in millimetres and every interval is in seconds; both derive from
    the paper scale, so an ungridded image cannot yield either.
    """
    report = analyse(_beat_with_p_wave(), fs=360.0,
                     px_per_mm=None, from_image=True)
    for key in ("heart_rate", "rhythm", "pr_interval", "qrs_duration",
                "qt_interval", "qtc", "st_segment", "axis", "rr_interval",
                "sdnn", "rmssd"):
        assert report[key].value is None, key
        assert report[key].quality == UNAVAILABLE, key
        assert report[key].reason == "ECG grid not detected - no timebase"
        assert report["display"][key] == "—", key


def test_hrv_measurements_are_stored_in_seconds():
    """Every Measurement in the report uses seconds; only display converts."""
    report = analyse(_beat_with_p_wave(), fs=360.0)
    # Guarding on "is not None" would let the assertions vanish silently.
    assert report["sdnn"].value is not None
    # ~0.0 s for a metronomic synthetic signal, but certainly sub-second.
    assert report["sdnn"].value < 1.0
    assert report["display"]["sdnn"].endswith("ms")


def test_display_strings_use_dash_for_unavailable():
    report = analyse(np.zeros(500), fs=360.0)
    assert report["display"]["heart_rate"] == "—"


from ecg.digitize import _trace_row_band


def _ink_band(signal, h=140, amp_px=1.0, thickness=3):
    """Render a 1-D signal as an ink band, the way digitize sees a printed lead."""
    band = np.zeros((h, len(signal)))
    mid = h // 2
    for x, v in enumerate(signal):
        y = int(round(mid - v * amp_px))
        for t in range(thickness):
            yy = y + t - thickness // 2
            if 0 <= yy < h:
                band[yy, x] = 255.0
    return band


def test_axis_angle_reflects_lead_amplitude_ratio():
    """Digitized leads must share one gain, or the axis degrees are noise.

    Per-lead standard-deviation normalisation (the old _trace_row_band) throws
    away exactly the lead I : aVF amplitude ratio that arctan2 encodes, so it
    reports +45 degrees for ANY pair of same-shaped leads. Two ratios are
    checked so a single lucky value cannot pass.
    """
    beat = _beat_with_p_wave() / 3.0            # peak amplitude 1.0

    # Equal amplitudes -> +45 degrees.
    equal = {"I": _trace_row_band(_ink_band(beat, amp_px=30)),
             "aVF": _trace_row_band(_ink_band(beat, amp_px=30))}
    m = qrs_axis(equal, fs=360.0)
    assert m.value is not None
    assert abs(m.value - 45.0) < 5.0, m.value

    # aVF at half the amplitude of I -> arctan(0.5) = +26.6 degrees. This is
    # the assertion the per-lead-normalised digitizer cannot satisfy: it
    # returns +45 here too.
    ratio = {"I": _trace_row_band(_ink_band(beat, amp_px=30)),
             "aVF": _trace_row_band(_ink_band(beat, amp_px=15))}
    m = qrs_axis(ratio, fs=360.0)
    assert m.value is not None
    assert abs(m.value - 26.57) < 6.0, m.value
