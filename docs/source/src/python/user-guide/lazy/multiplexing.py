# --8<-- [start:setup]
import polars as pl
import numpy as np
import tempfile
import base64
import polars.testing


def show_plan(q: pl.LazyFrame, optimized: bool = True):
    with tempfile.NamedTemporaryFile() as fp:
        q.show_graph(show=False, output_path=fp.name, optimized=optimized)
        with open(fp.name, "rb") as f:
            png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')


# --8<-- [end:setup]


# --8<-- [start:dataframe]
np.random.seed(0)
a = np.arange(0, 10)
np.random.shuffle(a)
df = pl.DataFrame({"n": a})
print(df)
# --8<-- [end:dataframe]

# --8<-- [start:eager]
# A group-by doesn't guarantee order
df1 = df.group_by("n").len()

# Take the lower half and the upper half in a list
out = [df1.slice(offset=i * 5, length=5) for i in range(2)]

# Assert df1 is equal to the sum of both halves
pl.testing.assert_frame_equal(df1, pl.concat(out))
# --8<-- [end:eager]

"""
# --8<-- [start:lazy]
lf1 = df.lazy().group_by("n").len()

out = [lf1.slice(offset=i * 5, length=5).collect() for i in range(2)]

pl.testing.assert_frame_equal(lf1.collect(), pl.concat(out))
# --8<-- [end:lazy]
"""

# --8<-- [start:plan_0]
q1 = df.lazy().group_by("n").len()
show_plan(q1, optimized=False)
# --8<-- [end:plan_0]

# --8<-- [start:plan_1]
q1 = df.lazy().group_by("n").len()
q2 = q1.slice(offset=0, length=5)
show_plan(q2, optimized=False)
# --8<-- [end:plan_1]

# --8<-- [start:plan_2]
q1 = df.lazy().group_by("n").len()
q2 = q1.slice(offset=5, length=5)
show_plan(q2, optimized=False)
# --8<-- [end:plan_2]


# --8<-- [start:collect_all]
lf1 = df.lazy().group_by("n").len()

out = [lf1.slice(offset=i * 5, length=5) for i in range(2)]
results = pl.collect_all([lf1] + out)

pl.testing.assert_frame_equal(results[0], pl.concat(results[1:]))
# --8<-- [end:collect_all]

# --8<-- [start:explain_all]
lf1 = df.lazy().group_by("n").len()
out = [lf1.slice(offset=i * 5, length=5) for i in range(2)]

print(pl.explain_all([lf1] + out))
# --8<-- [end:explain_all]
