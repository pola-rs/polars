import matplotlib.pyplot as plt

with open("../data/python_bench.txt") as f:
    pandas = [int(a) for a in f.read().split("\n")[:-1]]
with open("../data/python_bench_str.txt") as f:
    pandas_str = [int(a) for a in f.read().split("\n")[:-1]]

with open("../data/rust_bench.txt") as f:
    polars = [int(a) for a in f.read().split("\n")[:-1]]
with open("../data/rust_bench_str.txt") as f:
    polars_str = [int(a) for a in f.read().split("\n")[:-1]]

sizes = [1e2, 1e3, 1e4, 1e5, 1e6]

plt.figure(figsize=(14, 6))
plt.plot(sizes, pandas, label="pandas", color="C0")
plt.plot(sizes, polars, label="polars", color="C1")
plt.plot(sizes, pandas_str, label="pandas_str", color="C0", ls="-.")
plt.plot(sizes, polars_str, label="polars_str", color="C1", ls="-.")
plt.legend()
plt.title("Group by on 10 groups")
plt.ylabel("time [microseconds]")
plt.xscale("log")
plt.xlabel("dataset size")
plt.savefig("img/groupby10_.png")
