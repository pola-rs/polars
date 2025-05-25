import time
import polars as pl
from polars.datatypes import Struct

# pytest py-polars/tests/unit/expr/test_struct.py -k test_deeply_nested_structs_does_not_hang -s
def test_deeply_nested_structs_does_not_hang():
    df = pl.select(x=1)
    expr = pl.struct("x")
    for _ in range(21):
        expr = pl.struct(expr)

    start = time.perf_counter()
    out = df.select(expr)
    duration = time.perf_counter() - start

    assert isinstance(out.schema[out.columns[0]], Struct)
    assert duration < 5, f"Execution took too long: {duration:.2f} seconds"

# pytest py-polars/tests/unit/expr/test_struct.py -k test_struct_nesting_performance_benchmark -s
def test_struct_nesting_performance_benchmark():
    for i in range(10, 21):
        expr = pl.struct("x")
        for _ in range(i):
            expr = pl.struct(expr)
        df = pl.select(x=1)
        optimized = df.lazy().select(expr).explain(optimized=True)
        print(f"Depth {i} → plan length: {len(optimized)}")

# pytest py-polars/tests/unit/expr/test_struct.py -k test_struct_nesting_scalability
def test_struct_nesting_scalability():
    results = []
    for i in range(10, 21):
        expr = pl.struct("x")
        for _ in range(i):
            expr = pl.struct(expr)

        start = time.perf_counter()
        _ = pl.select(x=1).select(expr)
        duration = time.perf_counter() - start
        results.append((i, duration))

    print("\nDepth\tSeconds")
    for depth, sec in results:
        print(f"{depth}\t{sec:.2f}")

    assert results[-1][1] < 5, f"At depth {results[-1][0]} it took {results[-1][1]:.2f} seconds"
