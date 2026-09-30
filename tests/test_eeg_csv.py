"""CSV EEG: one column per channel, a stated sampling rate, microvolts."""
import numpy as np
import pytest

from eeg.loaders import RecordingError, read_csv_recording

FS = 128.0


def _write(tmp_path, rows, header=None, name="eeg.csv"):
    p = tmp_path / name
    lines = [",".join(header)] if header else []
    lines += [",".join(f"{v:.6f}" for v in r) for r in rows]
    p.write_text("\n".join(lines) + "\n")
    return str(p)


def _channels(seconds=10.0, freqs=(10.0, 6.0)):
    t = np.arange(0, seconds, 1 / FS)
    return t, np.stack([20 * np.sin(2 * np.pi * f * t) for f in freqs])


def test_columns_are_channels_with_header_names(tmp_path):
    _t, sig = _channels()
    path = _write(tmp_path, sig.T, header=["Fp1", "O1"])
    data, fs, names = read_csv_recording(path, fs=FS)
    assert data.shape == sig.shape and fs == FS and names == ["Fp1", "O1"]


def test_headerless_file_gets_generic_names(tmp_path):
    _t, sig = _channels()
    data, _fs, names = read_csv_recording(_write(tmp_path, sig.T), fs=FS)
    assert names == ["ch1", "ch2"] and data.shape[0] == 2


def test_a_time_column_is_dropped(tmp_path):
    t, sig = _channels()
    path = _write(tmp_path, np.column_stack([t, sig.T]), header=["time", "Fp1", "O1"])
    data, _fs, names = read_csv_recording(path, fs=FS)
    assert names == ["Fp1", "O1"]


def test_band_pass_keeps_the_rhythm(tmp_path):
    from eeg.metrics import power_spectrum
    _t, sig = _channels(freqs=(10.0,))
    data, fs, _ = read_csv_recording(_write(tmp_path, (sig + 300.0).T), fs=FS)
    f, p = power_spectrum(data[0], fs)
    assert f[int(np.argmax(p))] == pytest.approx(10.0, abs=0.6)
    assert abs(float(np.mean(data[0]))) < 5.0, "DC offset must be filtered out"


def test_sampling_rate_is_required(tmp_path):
    _t, sig = _channels()
    with pytest.raises(RecordingError, match="sampling rate"):
        read_csv_recording(_write(tmp_path, sig.T), fs=None)


def test_non_numeric_cells_are_refused(tmp_path):
    p = tmp_path / "bad.csv"
    p.write_text("Fp1,O1\n1.0,2.0\nabc,3.0\n")
    with pytest.raises(RecordingError, match="non-numeric"):
        read_csv_recording(str(p), fs=FS)


def test_an_empty_file_is_refused(tmp_path):
    p = tmp_path / "empty.csv"
    p.write_text("")
    with pytest.raises(RecordingError):
        read_csv_recording(str(p), fs=FS)
