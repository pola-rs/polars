from datetime import datetime

from polars.utils import in_nanoseconds_window


def test_in_ns_window() -> None:
    assert not in_nanoseconds_window(datetime(year=2600, month=1, day=1))
    assert in_nanoseconds_window(datetime(year=2000, month=1, day=1))
