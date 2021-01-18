import datetime
import glob
import pypolars as pl
from pypolars.datatypes import Utf8
from cmp.utils import peak_memory

files = glob.glob("../data/1*.csv")
files.sort()

with open("../data/mem_polars.txt", "w") as mem_f:
    with open("../data/pypolars_bench.txt", "w") as fh:
        with open("../data/pypolars_bench_str.txt", "w") as f_str:
            for fn in files:
                df = pl.read_csv(fn)
                df["str"] = df["str"].cast(Utf8)

                t0 = datetime.datetime.now()
                for _ in range(3):
                    res = df.groupby("groups").select("values").sum()
                duration = (datetime.datetime.now() - t0) / 3
                fh.write(f"{duration.microseconds}\n")
                t0 = datetime.datetime.now()
                for _ in range(3):
                    res = df.groupby("str").select("values").sum()
                duration = (datetime.datetime.now() - t0) / 3
                f_str.write(f"{duration.microseconds}\n")
                mem_f.write(str(peak_memory()) + "\n")
