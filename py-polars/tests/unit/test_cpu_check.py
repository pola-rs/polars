from unittest.mock import Mock

import pytest

from polars import _cpu_check
from polars._cpu_check import check_cpu_flags


@pytest.fixture()
def _feature_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    """Use the default set of feature flags."""
    feature_flags = "+sse3,+ssse3"
    monkeypatch.setattr(_cpu_check, "_POLARS_FEATURE_FLAGS", feature_flags)


@pytest.mark.usefixtures("_feature_flags")
def test_check_cpu_flags(
    monkeypatch: pytest.MonkeyPatch, recwarn: pytest.WarningsRecorder
) -> None:
    cpu_flags = {"sse3": True, "ssse3": True}
    mock_read_cpu_flags = Mock(return_value=cpu_flags)
    monkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags()

    assert len(recwarn) == 0


@pytest.mark.usefixtures("_feature_flags")
def test_check_cpu_flags_missing_features(monkeypatch: pytest.MonkeyPatch) -> None:
    cpu_flags = {"sse3": True, "ssse3": False}
    mock_read_cpu_flags = Mock(return_value=cpu_flags)
    monkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    with pytest.warns(RuntimeWarning, match="Missing required CPU features") as w:
        check_cpu_flags()

    assert "ssse3" in str(w[0].message)


def test_check_cpu_flags_unknown_flag(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    real_cpu_flags = {"sse3": True, "ssse3": False}
    mock_read_cpu_flags = Mock(return_value=real_cpu_flags)
    monkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)
    unknown_feature_flags = "+sse3,+ssse3,+HelloWorld!"
    monkeypatch.setattr(_cpu_check, "_POLARS_FEATURE_FLAGS", unknown_feature_flags)
    with pytest.raises(RuntimeError, match="unknown feature flag: 'HelloWorld!'"):
        check_cpu_flags()


def test_check_cpu_flags_skipped_no_flags(monkeypatch: pytest.MonkeyPatch) -> None:
    mock_read_cpu_flags = Mock()
    monkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags()

    assert mock_read_cpu_flags.call_count == 0


@pytest.mark.usefixtures("_feature_flags")
def test_check_cpu_flags_skipped_lts_cpu(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(_cpu_check, "_POLARS_LTS_CPU", True)

    mock_read_cpu_flags = Mock()
    monkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags()

    assert mock_read_cpu_flags.call_count == 0


@pytest.mark.usefixtures("_feature_flags")
def test_check_cpu_flags_skipped_env_var(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("POLARS_SKIP_CPU_CHECK", "1")

    mock_read_cpu_flags = Mock()
    monkeypatch.setattr(_cpu_check, "_read_cpu_flags", mock_read_cpu_flags)

    check_cpu_flags()

    assert mock_read_cpu_flags.call_count == 0
