import datetime
import pandas as pd
import glob
from cmp.utils import peak_memory

files = glob.glob("../data/1*.csv")
files.sort()

with open("../data/mem_pandas.txt", "w") as mem_f:
    with open("../data/python_bench.txt", "w") as f:
        with open("../data/python_bench_str.txt", "w") as f_str:

            for fn in files:
                df = pd.read_csv(fn)
                df = df.astype({"str": "str"})

                t0 = datetime.datetime.now()
                for _ in range(3):
                    res = df.groupby("groups").sum()
                duration = (datetime.datetime.now() - t0) / 3
                f.write(f"{duration.microseconds}\n")

                df = df[["str", "values"]]
                t0 = datetime.datetime.now()
                for _ in range(3):
                    res = df.groupby("str").sum()
                duration = (datetime.datetime.now() - t0) / 3
                f_str.write(f"{duration.microseconds}\n")
                mem_f.write(str(peak_memory()) + "\n")
