import polars as pl
from tests.unit.conftest import skip_wasm_differences


@skip_wasm_differences
def test_hash_struct() -> None:
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    df = df.select(pl.struct(pl.all()))
    assert df.select(pl.col("a").hash())["a"].to_list() == [
        5535262844797696299,
        15139341575481673729,
        12593759486533989774,
    ]
