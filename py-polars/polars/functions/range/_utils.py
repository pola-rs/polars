from __future__ import annotations

from datetime import timedelta

from polars.utils.convert import parse_duration_input


def parse_interval_argument(interval: str | timedelta) -> str:
    """Parse the interval argument as a Polars duration string."""
    if isinstance(interval, timedelta):
        return parse_duration_input(interval)

    if " " in interval:
        interval = interval.replace(" ", "")
    return interval.lower()
