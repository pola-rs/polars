import polars as pl
from polars.testing import assert_series_equal


def test_list_unique_boolean_22753() -> None:
    s = pl.Series(
        [
            [],
            [None],
            [None, None],
            [None, None, None],
            [None, None, None, False],
            [None, None, None, False, None],
            [None, None, None, False, None],
            [None, None, None, False, None, None, None],
            [None, None, None, False, None, None, None],
            [None, None, None, False, None, None, None, True, True],
        ]
    )

    assert_series_equal(
        s.list.unique(),
        pl.Series(
            [
                [],
                [None],
                [None],
                [None],
                [False, None],
                [False, None],
                [False, None],
                [False, None],
                [False, None],
                [False, True, None],
            ]
        ),
    )
