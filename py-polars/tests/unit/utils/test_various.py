import pytest

from polars._utils.various import deduplicate_names, parse_version
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


def test_deduplicate_names_avoids_colliding_with_an_existing_name() -> None:
    # The generated suffix must not reuse a name that the input already carries.
    assert deduplicate_names(["a", "a", "a0"]) == ["a", "a0", "a00"]
    assert deduplicate_names(["x", "x0", "x"]) == ["x", "x0", "x1"]
    assert deduplicate_names(["a", "a", "a", "a1"]) == ["a", "a0", "a1", "a10"]


def test_deduplicate_names_is_unchanged_without_collisions() -> None:
    assert deduplicate_names(["a", "b", "c"]) == ["a", "b", "c"]
    assert deduplicate_names(["a", "a", "a"]) == ["a", "a0", "a1"]
    assert deduplicate_names([]) == []
