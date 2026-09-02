# hello
import polars as pl

df = pl.DataFrame({"a": ["a0", "a1"], "b": ["b0", "b1"]}).lazy()

q = df.select(pl.col("a"), pl.col("b").reverse())
print(q.explain())
q.show_graph(plan_stage="physical", engine="streaming")
