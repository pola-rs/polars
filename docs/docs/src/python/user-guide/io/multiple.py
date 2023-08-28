# --8<-- [start:create]
import polars as pl

df = pl.DataFrame({"foo": [1, 2, 3], "bar": [None, "ham", "spam"]})

for i in range(5):
    df.write_csv(f"my_many_files_{i}.csv")
# --8<-- [end:create]

# --8<-- [start:read]
df = pl.read_csv("my_many_files_*.csv")
print(df)
# --8<-- [end:read]

# --8<-- [start:creategraph]
import base64

pl.scan_csv("my_many_files_*.csv").show_graph(
    output_path="images/multiple.png", show=False
)
with open("images/multiple.png", "rb") as f:
    png = base64.b64encode(f.read()).decode()
    print(f'<img src="data:image/png;base64, {png}"/>')
# --8<-- [end:creategraph]

# --8<-- [start:graph]
pl.scan_csv("my_many_files_*.csv").show_graph()
# --8<-- [end:graph]

# --8<-- [start:glob]
import polars as pl
import glob

queries = []
for file in glob.glob("my_many_files_*.csv"):
    q = pl.scan_csv(file).groupby("bar").agg([pl.count(), pl.sum("foo")])
    queries.append(q)

dataframes = pl.collect_all(queries)
print(dataframes)
# --8<-- [end:glob]
