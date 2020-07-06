import pandas as pd
import datetime

left = pd.read_csv("../data/join_left_80000.csv")
right = pd.read_csv("../data/join_right_80000.csv")

t0 = datetime.datetime.now()
joined = left.merge(right, on="key", how="inner")
duration = datetime.datetime.now() - t0
print("inner join {} μs".format(duration.microseconds))
print("shape:", joined.shape)
t0 = datetime.datetime.now()
joined = left.merge(right, on="key", how="left")
duration = datetime.datetime.now() - t0
print("left join {} μs".format(duration.microseconds))
print("shape:", joined.shape)
t0 = datetime.datetime.now()
joined = left.merge(right, on="key", how="outer")
duration = datetime.datetime.now() - t0
print("outer join {} μs".format(duration.microseconds))
print("shape:", joined.shape)
