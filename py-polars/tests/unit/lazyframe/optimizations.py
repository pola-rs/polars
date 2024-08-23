import polars as pl
from polars.testing import assert_frame_equal


def test_remove_double_sort() -> None:
    assert (
        pl.LazyFrame({"a": [1, 2, 3, 3]}).sort("a").sort("a").explain().count("SORT")
        == 1
    )

def test_in_null_all_optimization() -> None:
    lf = pl.LazyFrame({'group': [0, 0, 0, 1], 'val': [6, 0, None, None]})

    expected_df = pl.DataFrame({'group': [0, 1], 'val': [False, True]})
    result_df = lf.group_by('group').agg((pl.col('val').is_null().all())).collect()

    assert_frame_equal(expected_df, result_df)