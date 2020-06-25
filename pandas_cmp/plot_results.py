import matplotlib.pyplot as plt

with open("../data/python_bench.txt") as f:
    pandas = [int(a) for a in f.read().split("\n")[:-1]]

with open("../data/rust_bench.txt") as f:
    polars = [int(a) for a in f.read().split("\n")[:-1]]

sizes = [1e2, 1e3, 1e4, 1e5, 1e6]

plt.figure(figsize=(14, 6))
plt.plot(sizes, pandas, label="pandas")
plt.plot(sizes, polars, label="polars")
plt.legend()
plt.title("Group by on 10 groups")
plt.ylabel("time [microseconds]")
plt.xscale("log")
plt.xlabel("dataset size")
plt.savefig("img/groupby10.png")
