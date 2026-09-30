"""Compare two studies of the same modality. Pure: no I/O."""
from records.catalog import entries, lookup, range_status


def _fmt(value, unit):
    if value is None:
        return "not measured"
    text = f"{value:.1f}".rstrip("0").rstrip(".") if isinstance(value, float) else str(value)
    return f"{text} {unit}".strip()


def compare(before, after):
    """Rows describing what changed from `before` to `after`.

    Each row: key, label, unit, before, after, delta, direction ('up',
    'down', 'same' or None), changed (bool), note. The first rows are the
    verdict and headline; measurement rows follow in catalog order, then any
    extra keys either study carries.
    """
    if before["modality"] != after["modality"]:
        raise ValueError("Only studies of the same modality can be compared.")
    modality = after["modality"]
    rows = []
    for key, label in (("verdict", "Overall verdict"), ("headline", "Finding")):
        b, a = before.get(key), after.get(key)
        rows.append({"key": key, "label": label, "unit": "", "before": b,
                     "after": a, "delta": None, "direction": None,
                     "changed": b != a,
                     "note": f"changed from {b} to {a}" if b != a else "unchanged"})

    keys = [k for k, *_ in entries(modality)]
    keys += sorted((set(before["measurements"]) | set(after["measurements"])) - set(keys))
    for key in keys:
        mb = before["measurements"].get(key)
        ma = after["measurements"].get(key)
        if mb is None and ma is None:
            continue
        label, unit, rng = lookup(modality, key)
        vb = mb["value"] if mb else None
        va = ma["value"] if ma else None
        unit = unit or (ma or mb).get("unit") or ""
        delta = direction = None
        if vb is not None and va is not None:
            delta = va - vb
            direction = "same" if abs(delta) < 1e-9 else ("up" if delta > 0 else "down")
        sb, sa = range_status(vb, rng), range_status(va, rng)
        if va is None:
            note = "not measured this time"
        elif vb is None:
            note = "not measured before"
        elif sa and sa != sb:
            note = f"now {sa}" + (f" (was {sb})" if sb else "")
        elif sa:
            note = f"{sa}, unchanged range"
        else:
            note = ""
        rows.append({"key": key, "label": label, "unit": unit,
                     "before": _fmt(vb, unit), "after": _fmt(va, unit),
                     "before_value": vb, "after_value": va,
                     "delta": delta, "direction": direction,
                     "changed": direction not in (None, "same") or sa != sb,
                     "note": note})
    return rows
