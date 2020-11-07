import matplotlib.pyplot as plt
import numpy as np


def parse_result(f):
    return [float(a) / 1000 for a in f.read().split("\n")[:-1]]


with open("../data/python_bench.txt") as f:
    pandas = parse_result(f)
with open("../data/python_bench_str.txt") as f:
    pandas_str = parse_result(f)

with open("../data/pypolars_bench.txt") as f:
    pypolars = parse_result(f)
with open("../data/pypolars_bench_str.txt") as f:
    pypolars_str = parse_result(f)

with open("../data/datatable_bench.txt") as f:
    datatable = parse_result(f)
with open("../data/datatable_bench_str.txt") as f:
    datatable_str = parse_result(f)

sizes = [1e4, 1e5, 1e6, 1e7]
lib = ["py-polars", "pydatatable", "pandas"]
x = np.arange(1, 4)

fig, ax = plt.subplots(1, len(sizes), figsize=(14, 4))
plt.suptitle("Group by on 10 groups")
plt.subplots_adjust(wspace=0.4)
r = 0
ax = ax[None, :]
for i in range(len(pypolars)):
    c = i
    ca = ax[r, c]

    ca.set_title(f"{int(sizes[i]):,} rows")
    ca.bar(
        x - 0.25 / 2,
        [pypolars[i], datatable[i], pandas[i]],
        color=["C0", "C1", "C2"],
        width=0.25,
        label="int",
    )
    ca.bar(
        x + 0.25 / 2,
        [pypolars_str[i], datatable_str[i], pandas_str[i]],
        color=["C0", "C1", "C2"],
        width=0.25,
        alpha=0.5,
        label="str",
    )
    ca.set_xticks(x)
    ca.set_xticklabels(lib)
    ca.set_ylabel("duration [seconds]")
    ca.legend()
plt.savefig("img/groupby10_.png")

with open("../data/mem_pandas.txt") as f:
    pandas = [float(a) for a in f.read().split("\n")[:-1]]
with open("../data/mem_datatable.txt") as f:
    datatable = [float(a) for a in f.read().split("\n")[:-1]]

with open("../data/mem_polars.txt") as f:
    pypolars = [float(a) for a in f.read().split("\n")[:-1]]

fig, ax = plt.subplots(1, len(sizes), figsize=(14, 4))
plt.suptitle("Memory usage during Groupby")
plt.subplots_adjust(wspace=0.5, bottom=0.2)
r = 0
ax = ax[None, :]
for i in range(len(pypolars)):
    c = i
    ca = ax[r, c]

    ca.set_title(f"{int(sizes[i]):,} rows")
    ca.bar(
        x,
        [pypolars[i], datatable[i], pandas[i]],
        color=["C0", "C1", "C2"],
        alpha=0.75,
        width=0.4,
    )
    ca.set_xticks(x)
    ca.set_xticklabels(lib, rotation=30)
    ca.set_ylabel("process memory [GB]")
plt.savefig("img/groupby10_mem.png")
