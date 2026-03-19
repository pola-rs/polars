import pytest

from polars.exceptions import (
    CategoricalRemappingWarning,
    ComputeError,
    CustomUFuncWarning,
    MapWithoutReturnDtypeWarning,
    OutOfBoundsError,
    PerformanceWarning,
    PolarsError,
    PolarsInefficientMapWarning,
    PolarsWarning,
)


def test_polars_error_base_class() -> None:
    msg = "msg"
    assert isinstance(ComputeError(msg), PolarsError)
    with pytest.raises(PolarsError, match=msg):
        raise OutOfBoundsError(msg)


def test_polars_warning_base_class() -> None:
    msg = "msg"
    assert isinstance(MapWithoutReturnDtypeWarning(msg), PolarsWarning)
    with pytest.raises(PolarsWarning, match=msg):
        raise CustomUFuncWarning(msg)


def test_performance_warning_base_class() -> None:
    msg = "msg"
    assert isinstance(PolarsInefficientMapWarning(msg), PerformanceWarning)
    with pytest.raises(PerformanceWarning, match=msg):
        raise CategoricalRemappingWarning(msg)
