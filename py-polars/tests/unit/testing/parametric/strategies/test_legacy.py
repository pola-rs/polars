import pytest
from hypothesis.errors import NonInteractiveExampleWarning

from polars.testing.parametric import columns, create_list_strategy
from polars.testing.parametric.strategies.core import _COL_LIMIT


@pytest.mark.hypothesis()
def test_columns_deprecated() -> None:
    with pytest.deprecated_call(), pytest.warns(NonInteractiveExampleWarning):
        result = columns()
    assert 0 <= len(result) <= _COL_LIMIT


@pytest.mark.hypothesis()
def test_create_list_strategy_deprecated() -> None:
    with pytest.deprecated_call(), pytest.warns(NonInteractiveExampleWarning):
        result = create_list_strategy(size=5)
    with pytest.warns(NonInteractiveExampleWarning):
        assert len(result.example()) == 5
