# GPU Support [Open Beta]

Polars provides an in-memory, GPU-accelerated execution engine for Python users of the Lazy API on NVIDIA GPUs using [RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/). This functionality is available in Open Beta and is undergoing rapid development.

### System Requirements

- NVIDIA Volta™ or higher GPU with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
- CUDA 11 or CUDA 12
- Linux or Windows Subsystem for Linux 2 (WSL2)

See the [RAPIDS installation guide](https://docs.rapids.ai/install#system-req) for full details.

### Installation

You can install the GPU backend for Polars with a feature flag as part of a normal [installation](installation.md).

=== ":fontawesome-brands-python: Python"

```bash
pip install --extra-index-url=https://pypi.nvidia.com polars[gpu]
```

!!! note Installation on a CUDA 11 system

    If you have CUDA 11, the installation line is slightly more complicated: the relevant GPU package must be requested by hand.

    === ":fontawesome-brands-python: Python"
    ```bash
    pip install --extra-index-url=https://pypi.nvidia.com polars cudf-polars-cu11
    ```

### Usage

Having built a query using the lazy API [as normal](lazy/index.md), GPU-enabled execution is requested by running `.collect(engine="gpu")` instead of `.collect()`.

{{ code_header("python", [], []) }}

```python
--8<-- "python/user-guide/lazy/gpu.py:setup"

result = q.collect(engine="gpu")
print(result)
```

```python exec="on" result="text" session="user-guide/lazy"
--8<-- "python/user-guide/lazy/gpu.py:setup"
--8<-- "python/user-guide/lazy/gpu.py:simple-result"
```

For more detailed control over the execution, for example to specify which GPU to use on a multi-GPU node, we can provide a `GPUEngine` object. By default, the GPU engine will use a configuration applicable to most use cases.

{{ code_header("python", [], []) }}

```python
--8<-- "python/user-guide/lazy/gpu.py:engine-setup"
result = q.collect(engine=pl.GPUEngine(device=1))
print(result)
```

```python exec="on" result="text" session="user-guide/lazy"
--8<-- "python/user-guide/lazy/gpu.py:engine-setup"
--8<-- "python/user-guide/lazy/gpu.py:engine-result"
```

### How It Works

When you use the GPU-accelerated engine, Polars creates and optimizes a query plan and dispatches to a [RAPIDS](https://rapids.ai/) cuDF-based physical execution engine to compute the results on NVIDIA GPUs. The final result is returned as a normal CPU-backed Polars dataframe.

### What's Supported on the GPU?

GPU support is currently in Open Beta and the engine is undergoing rapid development. The engine currently supports many, but not all, of the core expressions and data types.

Since expressions are composable, it's not feasible to list a full matrix of expressions supported on the GPU. Instead, we provide a list of the high-level categories of expressions and interfaces that are currently supported and not supported.

#### Supported

- LazyFrame API
- SQL API
- I/O from CSV, Parquet, ndjson, and in-memory CPU DataFrames.
- Operations on numeric, logical, string, and datetime types
- String processing
- Aggregations and grouped aggregations
- Joins
- Filters
- Missing data
- Concatenation

#### Not Supported

- Eager DataFrame API
- Streaming API
- Operations on categorical, struct, and list data types
- Rolling aggregations
- Time series resampling
- Timezones
- Folds
- User-defined functions
- JSON, Excel, and Database file formats

#### Did my query use the GPU?

The release of the GPU engine in Open Beta implies that we expect things to work well, but there are still some rough edges we're working on. In particular the full breadth of the Polars expression API is not yet supported. With fallback to the CPU, your query _should_ complete, but you might not observe any change in the time it takes to execute. There are two ways to get more information on whether the query ran on the GPU.

When running in verbose mode, any queries that cannot execute on the GPU will issue a `PerformanceWarning`:

{{ code_header("python", [], []) }}

```python
--8<-- "python/user-guide/lazy/gpu.py:fallback-setup"

with pl.Config() as cfg:
    cfg.set_verbose(True)
    result = q.collect(engine="gpu")

print(result)
```

```python exec="on" result="text" session="user-guide/lazy"
--8<-- "python/user-guide/lazy/gpu.py:fallback-setup"
print(
    "PerformanceWarning: Query execution with GPU not supported, reason: \n"
    "<class 'NotImplementedError'>: Grouped rolling window not implemented"
)
print("# some details elided")
print()
print(q.collect())
```

To disable fallback, and have the GPU engine raise an exception if a query is unsupported, we can pass an appropriately configured `GPUEngine` object:

{{ code_header("python", [], []) }}

```python
q.collect(engine=pl.GPUEngine(raise_on_fail=True))
```

```pytb
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/coder/third-party/polars/py-polars/polars/lazyframe/frame.py", line 2035, in collect
    return wrap_df(ldf.collect(callback))
polars.exceptions.ComputeError: 'cuda' conversion failed: NotImplementedError: Grouped rolling window not implemented
```

Currently, only the proximal cause of failure to execute on the GPU is reported, we plan to extend this functionality to report all unsupported operations for a query.

### Testing

The Polars and NVIDIA RAPIDS teams run comprehensive unit and integration tests to ensure that the GPU-accelerated Polars backend works smoothly.

The **full** Polars test suite is run on every commit made to the GPU engine, ensuring consistency of results.

The GPU engine currently passes 99.2% of the Polars unit tests with CPU fallback enabled. Without CPU fallback, the GPU engine passes 88.8% of the Polars unit tests. With fallback, there are approximately 100 failing tests: around 40 of these fail due to mismatching debug output; there are some cases where the GPU engine produces the a correct result but uses a different data type; the remainder are cases where we do not correctly determine that a query is unsupported and therefore fail at runtime, instead of falling back.

### When Should I Use a GPU?

Based on our benchmarking, you're most likely to observe speedups using the GPU engine when your workflow's profile is dominated by grouped aggregations and joins. In contrast I/O bound queries typically show similar performance on GPU and CPU. GPUs typically have less RAM than CPU systems, therefore very large datasets will fail due to out of memory errors. Based on our testing, raw datasets of 50-100 GiB fit (depending on the workflow) well with a GPU with 80GiB of memory.

### CPU-GPU Interoperability

Both the CPU and GPU engine use the Apache Arrow columnar memory specification, making it possible to quickly move data between the CPU and GPU. Additionally, files written by one engine can be read by the other engine.

When using GPU mode, your workflow won't fail if something isn't supported. When you run `collect(engine="gpu")`, the optimized query plan is inspected to see whether it can be executed on the GPU. If it can't, it will transparently fall back to the standard Polars engine and run on the CPU.

GPU execution is only available in the Lazy API, so materialized DataFrames will reside in CPU memory when the query execution finishes.

### Providing feedback

Please report issues, and missing features, on the Polars [issue tracker](https://github.com/pola-rs/polars/issues).
