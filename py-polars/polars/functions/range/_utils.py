from __future__ import annotations

from datetime import timedelta

from polars.utils.convert import _timedelta_to_pl_duration


def parse_interval_argument(interval: str | timedelta) -> str:
    """Parse the interval argument as a Polars duration string."""
    if isinstance(interval, timedelta):
        return _timedelta_to_pl_duration(interval)

    if " " in interval:
        interval = interval.replace(" ", "")
    return interval.lower()
