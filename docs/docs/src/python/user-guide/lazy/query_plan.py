# --8<-- [start:setup]
import polars as pl

# --8<-- [end:setup]

# --8<-- [start:plan]
q1 = (
    pl.scan_csv(f"docs/src/data/reddit.csv")
    .with_columns(pl.col("name").str.to_uppercase())
    .filter(pl.col("comment_karma") > 0)
)
# --8<-- [end:plan]

# --8<-- [start:createplan]
import base64

q1.show_graph(optimized=False, show=False, output_path="images/query_plan.png")
with open("images/query_plan.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:createplan]

"""
# --8<-- [start:showplan]
q1.show_graph(optimized=False)
# --8<-- [end:showplan]
"""

# --8<-- [start:describe]
q1.explain(optimized=False)
# --8<-- [end:describe]

# --8<-- [start:createplan2]
q1.show_graph(show=False, output_path="images/query_plan_optimized.png")
with open("images/query_plan_optimized.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:createplan2]

"""
# --8<-- [start:show]
q1.show_graph()
# --8<-- [end:show]
"""

# --8<-- [start:optimized]
q1.explain()
# --8<-- [end:optimized]
