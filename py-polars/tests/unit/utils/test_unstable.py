from __future__ import annotations

import pytest

from polars._utils.unstable import issue_unstable_warning, unstable
from polars.exceptions import UnstableWarning


def test_issue_unstable_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    msg = "`func` is considered unstable."
    expected = (
        msg
        + " It may be changed at any point without it being considered a breaking change."
    )
    with pytest.warns(UnstableWarning, match=expected):
        issue_unstable_warning(msg)


def test_issue_unstable_warning_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    msg = "This functionality is considered unstable."
    with pytest.warns(UnstableWarning, match=msg):
        issue_unstable_warning()


def test_issue_unstable_warning_setting_disabled(
    recwarn: pytest.WarningsRecorder,
) -> None:
    issue_unstable_warning()
    assert len(recwarn) == 0


def test_unstable_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    @unstable()
    def hello() -> None: ...

    msg = "`hello` is considered unstable."
    with pytest.warns(UnstableWarning, match=msg):
        hello()


def test_unstable_decorator_setting_disabled(recwarn: pytest.WarningsRecorder) -> None:
    @unstable()
    def hello() -> None: ...

    hello()
    assert len(recwarn) == 0
