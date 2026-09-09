from ecg.quality import Measurement, OK, UNAVAILABLE


def test_measurement_formats_value_with_unit():
    m = Measurement(value=0.156, quality=OK)
    assert m.format("s", decimals=3) == "0.156 s"


def test_unavailable_measurement_renders_dash():
    m = Measurement(value=None, quality=UNAVAILABLE, reason="P wave absent")
    assert m.format("ms") == "—"
    assert m.reason == "P wave absent"
