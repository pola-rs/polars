import base64

# --8<-- [start:import]
import polars as pl
# --8<-- [end:import]

# --8<-- [start:streaming]
q1 = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(pl.col("sepal_width").mean())
)
df = q1.collect(engine="streaming")
# --8<-- [end:streaming]

"""
# --8<-- [start:createplan_query]
q1 = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(
        mean_width=pl.col("sepal_width").mean(),
        mean_width2=pl.col("sepal_width").sum() / pl.col("sepal_length").count(),
    )
    .show_graph(plan_stage="physical", engine="streaming")
)
# --8<-- [end:createplan_query]
"""

# --8<-- [start:createplan]
import base64
import polars as pl

q1 = (
    pl.scan_csv("docs/assets/data/iris.csv")
    .filter(pl.col("sepal_length") > 5)
    .group_by("species")
    .agg(
        mean_width=pl.col("sepal_width").mean(),
        mean_width2=pl.col("sepal_width").sum() / pl.col("sepal_length").count(),
    )
)

q1.show_graph(
    plan_stage="physical",
    engine="streaming",
    show=False,
    output_path="docs/assets/images/query_plan.png",
)
with open("docs/assets/images/query_plan.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:createplan]
