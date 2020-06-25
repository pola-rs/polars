import datetime
import pandas as pd
import glob

files = glob.glob("../data/1*.csv")
files.sort()
with open("../data/python_bench.txt", 'w') as f:
    with open("../data/python_bench_str.txt", 'w') as f_str:

        for fn in files:
            df = pd.read_csv(fn)
            df = df.astype({'str': 'str'})
            t0 = datetime.datetime.now()
            res = df.groupby("groups").sum()
            duration = datetime.datetime.now() - t0
            print(fn, duration.microseconds, res)
            f.write(f"{duration.microseconds}\n")

            t0 = datetime.datetime.now()
            res = df.groupby("str").sum()
            duration = datetime.datetime.now() - t0
            f_str.write(f"{duration.microseconds}\n")
