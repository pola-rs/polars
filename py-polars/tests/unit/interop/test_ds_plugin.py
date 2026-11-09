from pathlib import Path

import pytest

import polars as pl
from polars._dependencies import _lazy_import
from polars.plugins import register_plugin_function
from polars.testing import assert_frame_equal

# don't import polars_ds until an actual test is triggered (the decorator already
# ensures the tests aren't run locally; this avoids premature local import)
pds, _ = _lazy_import("polars_ds")

pytestmark = pytest.mark.ci_only


def test_basic_operation() -> None:
    # We are mostly interested in making sure that we can actually still call the plugin
    # properly.
    df = pl.DataFrame({"name": ["a", "b", "c"]})
    assert_frame_equal(
        df.select(pds.str_leven("name", pl.lit("che"))),
        pl.Series("name", [3, 3, 2], pl.UInt32).to_frame(),
    )


def _levenshtein(is_deterministic: bool) -> pl.Expr:
    plugin_file = pds.__file__
    assert plugin_file is not None
    return register_plugin_function(
        plugin_path=Path(plugin_file).parent,
        function_name="pl_levenshtein",
        args=[pl.col("name"), pl.lit("che"), pl.lit(False)],
        is_deterministic=is_deterministic,
    )


@pytest.mark.parametrize("is_deterministic", [True, False])
def test_plugin_cspe(is_deterministic: bool) -> None:
    lf = pl.LazyFrame({"name": ["a", "b", "c"]})
    query = pl.concat([lf.select(_levenshtein(is_deterministic)) for _ in range(2)])
    optimizations = pl.QueryOptFlags(comm_subplan_elim=True, comm_subexpr_elim=False)

    plan = query.explain(optimizations=optimizations)
    assert plan.count("pl_levenshtein") == (1 if is_deterministic else 2)
    # Even a non-deterministic plugin can share its deterministic input.
    assert plan.count("CACHE[id:") == 2
    assert plan.count('DF ["name"]') == 1
    assert_frame_equal(
        query.collect(optimizations=optimizations),
        pl.Series("name", [3, 3, 2, 3, 3, 2], pl.UInt32).to_frame(),
    )


@pytest.mark.parametrize("is_deterministic", [True, False])
def test_plugin_cse(is_deterministic: bool) -> None:
    lf = pl.LazyFrame({"name": ["a", "b", "c"]})
    query = lf.select(
        _levenshtein(is_deterministic).alias("a"),
        _levenshtein(is_deterministic).alias("b"),
    )
    optimizations = pl.QueryOptFlags(comm_subexpr_elim=True)

    plan = query.explain(optimizations=optimizations)
    assert plan.count("pl_levenshtein") == (1 if is_deterministic else 2)
    assert ("__POLARS_CSER" in plan) is is_deterministic
    assert_frame_equal(
        query.collect(optimizations=optimizations),
        pl.DataFrame(
            {"a": [3, 3, 2], "b": [3, 3, 2]}, schema={"a": pl.UInt32, "b": pl.UInt32}
        ),
    )
