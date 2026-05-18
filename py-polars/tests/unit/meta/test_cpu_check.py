from __future__ import annotations

from typing import TYPE_CHECKING
from unittest.mock import Mock

import pytest

from polars import _cpu_check
from polars._cpu_check import check_cpu_flags

if TYPE_CHECKING:
    from tests.conftest import PlMonkeyPatch

TEST_FEATURE_FLAGS = "+sse3,+ssse3"


def test_check_cpu_flags(
    plmonkeypatch: PlMonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    cpu_flags = {"sse3": True, "ssse3": True}
    mock_read_cpu_flags = Mock(return_value=cpu_flags)
    plmonkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags(TEST_FEATURE_FLAGS)

    assert len(recwarn) == 0


def test_check_cpu_flags_missing_features(plmonkeypatch: PlMonkeyPatch) -> None:
    cpu_flags = {"sse3": True, "ssse3": False}
    mock_read_cpu_flags = Mock(return_value=cpu_flags)
    plmonkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    with pytest.warns(RuntimeWarning, match="Missing required CPU features") as w:
        check_cpu_flags(TEST_FEATURE_FLAGS)

    assert "ssse3" in str(w[0].message)


def test_check_cpu_flags_unknown_flag(
    plmonkeypatch: PlMonkeyPatch,
) -> None:
    real_cpu_flags = {"sse3": True, "ssse3": False}
    mock_read_cpu_flags = Mock(return_value=real_cpu_flags)
    unknown_feature_flags = "+sse3,+ssse3,+HelloWorld!"
    plmonkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)
    with pytest.raises(RuntimeError, match="unknown feature flag: 'HelloWorld!'"):
        check_cpu_flags(unknown_feature_flags)


def test_check_cpu_flags_skipped_no_flags(plmonkeypatch: PlMonkeyPatch) -> None:
    mock_read_cpu_flags = Mock()
    plmonkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags("")

    assert mock_read_cpu_flags.call_count == 0


def test_check_cpu_flags_skipped_env_var(plmonkeypatch: PlMonkeyPatch) -> None:
    plmonkeypatch.setenv("POLARS_SKIP_CPU_CHECK", "1")

    mock_read_cpu_flags = Mock()
    plmonkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags(TEST_FEATURE_FLAGS)

    assert mock_read_cpu_flags.call_count == 0
