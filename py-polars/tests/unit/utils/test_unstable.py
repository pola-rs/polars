from __future__ import annotations

import pytest

import polars as pl
from polars.utils.unstable import issue_unstable_warning, unstable


def test_issue_unstable_warning(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    msg = "unstable"
    with pytest.warns(pl.UnstableWarning, match=msg):
        issue_unstable_warning(msg)


def test_issue_unstable_warning_setting_disabled(
    recwarn: pytest.WarningsRecorder,
) -> None:
    issue_unstable_warning("unstable")
    assert len(recwarn) == 0


def test_unstable_decorator(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_WARN_UNSTABLE", "1")

    @unstable()
    def hello() -> None:
        ...

    msg = "`hello` is considered unstable. Its API and implementation are subject to change."
    with pytest.warns(pl.UnstableWarning, match=msg):
        hello()


def test_unstable_decorator_setting_disabled(recwarn: pytest.WarningsRecorder) -> None:
    @unstable()
    def hello() -> None:
        ...

    hello()
    assert len(recwarn) == 0
