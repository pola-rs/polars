import pandas as pd
import numpy as np

groups = np.arange(10)
str_groups = np.array(list("0123456789"))
np.random.seed(1)

for size in [1e2, 1e3, 1e4, 1e5, 1e6]:
    size = int(size)
    g = np.random.choice(groups, size)
    sg = np.random.choice(str_groups, size)
    v = np.random.randn(size)
    df = pd.DataFrame({"groups": g, "values": v, "str": sg})
    df.to_csv(f"../data/{size}.csv", index=False)

print("data created")