import pytest

from polars._utils.various import issue_warning
from polars.exceptions import PerformanceWarning


def test_issue_warning() -> None:
    msg = "hello"
    with pytest.warns(PerformanceWarning, match=msg):
        issue_warning(msg, PerformanceWarning)
