import datetime
import glob
import datatable as dt
from datatable import f, by
import pandas as pd
from cmp.utils import peak_memory

files = glob.glob("../data/1*.csv")
files.sort()

with open("../data/mem_datatable.txt", "w") as mem_f:
    with open("../data/datatable_bench.txt", "w") as fh:
        with open("../data/datatable_bench_str.txt", "w") as f_str:
            for fn in files:
                df = pd.read_csv(fn)
                df = df.astype({"str": "str"})
                df = dt.Frame(df)

                t0 = datetime.datetime.now()
                for _ in range(3):
                    res = df[:, dt.sum(f.values), by(f.groups)]
                duration = (datetime.datetime.now() - t0) / 3
                fh.write(f"{duration.microseconds}\n")
                t0 = datetime.datetime.now()
                for _ in range(3):
                    res = df[:, dt.sum(f.values), by(f.str)]
                duration = (datetime.datetime.now() - t0) / 3
                f_str.write(f"{duration.microseconds}\n")
                mem_f.write(str(peak_memory()) + "\n")
