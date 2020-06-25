import datetime
import pandas as pd
import glob

files = glob.glob("../data/1*.csv")
files.sort()
with open("../data/python_bench.txt", 'w') as f:

    for fn in files:
        df = pd.read_csv(fn)
        t0 = datetime.datetime.now()
        res = df.groupby("groups").sum()
        duration = datetime.datetime.now() - t0
        print(fn, duration.microseconds, res)
        f.write(f"{duration.microseconds}\n")
