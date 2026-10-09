import pytest

from polars._utils.various import parse_version
from polars._warnings import issue_warning
from polars.exceptions import PerformanceWarning


def test_issue_warning() -> None:
    msg = "hello"
    with pytest.warns(PerformanceWarning, match=msg):
        issue_warning(msg, PerformanceWarning)


def test_parse_version_stops_at_the_first_non_digit() -> None:
    # A pre-release suffix must not be folded into the number that precedes it.
    assert parse_version("1.2.3rc1") == (1, 2, 3)
    assert parse_version("2.0.0rc2") < parse_version("2.0.1")
    assert parse_version("1.2.3rc1") < parse_version("1.2.4")


def test_parse_version_handles_a_component_without_digits() -> None:
    assert parse_version("1.0.dev") == (1, 0, 0)


def test_parse_version_is_unchanged_for_plain_versions() -> None:
    assert parse_version("1.2.3") == (1, 2, 3)
    assert parse_version("2026.7.0") == (2026, 7, 0)
    assert parse_version("v1.2.3") == (1, 2, 3)
    assert parse_version((1, 2, 3)) == (1, 2, 3)
