from __future__ import annotations

import subprocess
import sys


def test_global_categories_gc() -> None:
    # We run this in a subprocess to ensure no other test is or has been messing
    # with the global categories.
    out = subprocess.check_output(
        [
            sys.executable,
            "-c",
            """\
import polars as pl

df = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c"}
df2 = pl.DataFrame({"x": ["d", "e", "f"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c", "d", "e", "f"}
del df
del df2
df3 = pl.DataFrame({"x": ["x"]}, schema={"x": pl.Categorical})
assert set(df3["x"].cat.get_categories().to_list()) == {"x"}
del df3

keep_alive = pl.DataFrame({"x": []}, schema={"x": pl.Categorical})
df = pl.DataFrame({"x": ["a", "b", "c"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c"}
df2 = pl.DataFrame({"x": ["d", "e", "f"]}, schema={"x": pl.Categorical})
assert set(df["x"].cat.get_categories().to_list()) == {"a", "b", "c", "d", "e", "f"}
del df
del df2
df3 = pl.DataFrame({"x": ["x"]}, schema={"x": pl.Categorical})
assert set(df3["x"].cat.get_categories().to_list()) == {"a", "b", "c", "d", "e", "f", "x"}

print("OK", end="")
""",
        ],
    )

    assert out == b"OK"
