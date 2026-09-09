"""Measured values that can honestly report their own reliability."""
from dataclasses import dataclass

OK = "ok"
LOW = "low_confidence"
UNAVAILABLE = "unavailable"


@dataclass
class Measurement:
    value: float | None
    quality: str
    reason: str = ""

    def format(self, unit: str, decimals: int = 1) -> str:
        if self.value is None:
            return "—"
        return f"{self.value:.{decimals}f} {unit}"
