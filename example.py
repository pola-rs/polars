import polars as pl
import numpy as np
import time

# 1. Generate a large Series of random-length integer lists
N = 100_000
rng = np.random.default_rng(42)

def random_list():
    length = rng.integers(5, 20)
    lst = rng.integers(0, 10, size=length).tolist()
    lst.append(None)
    return lst

l = [random_list() for _ in range(N)]
s = pl.Series("a", l)
print(s)

# 2. Define the two operations
def using_eval(series: pl.Series) -> pl.Series:
    # the “current” approach: eval + element().filter(...)
    return series.list.eval(pl.element().filter(pl.element() < 5))

def using_filter(series: pl.Series) -> pl.Series:
    # the proposed new syntax
    return series.list.filter(pl.element() < 5)

# 3. Warm up (compile expressions)
_ = using_eval(s)
_ = using_filter(s)

# 4. Time them
reps = 5

t0 = time.perf_counter()
for _ in range(reps):
    _ = using_eval(s)
t_eval = (time.perf_counter() - t0) / reps

t0 = time.perf_counter()
for _ in range(reps):
    _ = using_filter(s)
t_filter = (time.perf_counter() - t0) / reps

print(f"Average time over {reps} runs:")
print(f"  list.eval(...filter...) : {t_eval:.4f} seconds")
print(f"  list.filter(...)         : {t_filter:.4f} seconds")
