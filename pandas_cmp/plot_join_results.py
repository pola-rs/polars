import matplotlib.pyplot as plt
import pandas as pd

with open("../data/python_bench_join.txt") as f:
    pandas = [float(a) for a in f.read().split("\n")[:-1]]

with open("../data/rust_bench_join.txt") as f:
    polars = [float(a) for a in f.read().split("\n")[:-1]]

x = ["inner", "left", "outer"]

df = pd.DataFrame({"join_type": x, "pandas": pandas, "polars": polars})

df.plot.bar(x="join_type", figsize=(14, 6))
plt.title("join on 80,000 rows")
plt.ylabel("time [μs]")
plt.savefig("img/join_80_000.png")
