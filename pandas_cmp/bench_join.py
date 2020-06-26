import pandas as pd
import datetime

df = pd.read_csv("../data/1000.csv")
df = df.astype({'str': 'str'})
size = 500
a = df[:size]
b = df[size:]

t0 = datetime.datetime.now()
joined = a.merge(b, on="groups", how="inner")
duration = datetime.datetime.now() - t0
print(duration.microseconds)
print(joined.shape)
