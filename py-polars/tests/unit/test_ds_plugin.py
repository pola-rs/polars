import pytest

import polars as pl
from polars.dependencies import _lazy_import
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
