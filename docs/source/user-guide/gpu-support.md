# GPU Support [Open Beta]

Polars provides a GPU-accelerated execution engine for Python users of the Lazy API on NVIDIA GPUs
using [RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/cudf_polars/). This functionality is
available in Open Beta and is undergoing rapid development.

### System Requirements

- NVIDIA Volta™ or higher GPU with [compute capability](https://developer.nvidia.com/cuda-gpus) 7.0+
- CUDA 12 or CUDA 13
- Linux or Windows Subsystem for Linux 2 (WSL2)

See the [RAPIDS installation guide](https://docs.rapids.ai/install#system-req) for the most
up-to-date details.

### Installation

You can install the GPU backend for Polars with a feature flag as part of a normal
[installation](installation.md).

=== ":fontawesome-brands-python: Python"

```bash
pip install polars[gpu]
```

!!! note

    `pip install polars[gpu]` is currently configured to install cudf-polars-cu12 for CUDA 12.
    If your system has a different version of CUDA, separately install the cudf-polars library
    with a suffix that matches your CUDA version.

    For example with CUDA 13:

    === ":fontawesome-brands-python: Python"
    ```bash
    pip install polars cudf-polars-cu13
    ```

### Usage

Having built a query using the lazy API [as normal](lazy/index.md), GPU-enabled execution is
requested by passing `engine="gpu"` to `.collect` or `.sink_*` calls.

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

#### `GPUEngine`s and multiple GPUs

`engine="gpu"` is convenient for running a query on a single GPU. Multi-GPU execution and other
query-runtime configurations can be specified by passing a `GPUEngine` object.

As of cudf-polars version 26.06, 3 `GPUEngine` subclasses are provided by cudf-polars library:

- [`RayEngine`](https://docs.rapids.ai/api/cudf/stable/cudf_polars/usage/#configuring-rayengine):
  Facilitates multi-GPU execution using [Ray](https://www.ray.io/)
- [`DaskEngine`](https://docs.rapids.ai/api/cudf/stable/cudf_polars/dask_engine/): Facilitates
  multi-GPU execution using [Dask](https://www.dask.org/)
- [`SMPDEngine`](https://docs.rapids.ai/api/cudf/stable/cudf_polars/spmd_engine/): Single program,
  multiple data model for multi-GPU execution

These 3 engines spin up resources that can be torn down by using the engines as context managers.

{{ code_header("python", [], []) }}

```python
--8<-- "python/user-guide/lazy/gpu.py:engine-setup"
from cudf_polars.engine.ray import RayEngine

with RayEngine() as engine:
    result = q.collect(engine=engine)
    print(result)
```

```python exec="on" result="text" session="user-guide/lazy"
--8<-- "python/user-guide/lazy/gpu.py:engine-setup"
--8<-- "python/user-guide/lazy/gpu.py:engine-result"
```

!!! note

    To use `RayEngine` or `DaskEngine`, Ray or Dask must be installed respectively. These
    dependencies can be installed with the `[ray]` or `[dask]` pip extra of cudf-polars.

    For example with CUDA 13:

    === ":fontawesome-brands-python: Python"
    ```bash
    pip install cudf-polars-cu13[ray]
    pip install cudf-polars-cu13[dask]
    ```

### How It Works

When you use the GPU-accelerated engine, Polars creates and optimizes a query plan and dispatches to
a [RAPIDS](https://rapids.ai/) cuDF-based physical execution engine to compute the results on NVIDIA
GPUs. The final result is returned as a normal CPU-backed Polars dataframe.

#### CPU-GPU Interoperability

Both the CPU and GPU engine use the Apache Arrow columnar memory specification, making it possible
to quickly move data between the CPU and GPU. Additionally, files written by one engine can be read
by the other engine.

When using GPU mode, your query won't fail if an operation isn't supported. When you pass
`engine="gpu"`, the optimized query plan is inspected to see whether it can be executed on the GPU.
If it can't, it will transparently fall back to the standard Polars engine and run all query
operations on the CPU.

GPU execution is only available in the Lazy API, so materialized DataFrames will reside in CPU
memory when the query execution finishes.

### What's Supported on the GPU?

GPU support is currently in Open Beta and the engine is undergoing rapid development. The engine
currently supports many, but not all, of the core expressions and data types.

Since expressions are composable, it's not feasible to list a full matrix of expressions supported
on the GPU. Instead, we provide a list of the high-level categories of expressions and interfaces
that are currently supported and not supported.

#### Supported

- LazyFrame API
- SQL API
- I/O from CSV, Parquet, ndjson, and in-memory CPU DataFrames.
- Operations on numeric, logical, string, and datetime types
- String processing
- Aggregations including grouped and rolling variants
- Joins
- Filters
- Missing data
- Concatenation

#### Not Supported

- Eager DataFrame API
- Date, Categorical, Enum, Time, Array, Binary and Object data types
- Some expressions for Datetime with Timezone and List types
- Time series resampling
- Folds
- User-defined functions
- Excel and Database file formats

#### Did my query use the GPU?

The Open Beta GPU engine implies that a significant portion of common, expression APIs are
supported, but a query containing an unsupported expression API will fallback to the CPU and not
observe any change in the time it takes to execute. There are two ways to get more information on
whether the query ran on the GPU.

When running in verbose mode, any queries that cannot execute on the GPU will issue a
`PerformanceWarning`:

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
    "PerformanceWarning: Query execution with GPU not possible: unsupported operations"
)
print("The errors were:")
print("- NotImplementedError: anonymousfunction")
print("  return wrap_df(ldf.collect(engine, callback))")
print()
print(q.collect())
```

To disable fallback, and have the GPU engine raise an exception if a query is unsupported, pass
`raise_on_fail=True` to a `GPUEngine` object:

{{ code_header("python", [], []) }}

```python
q.collect(engine=pl.GPUEngine(raise_on_fail=True))
```

```pytb
Traceback (most recent call last):
  File "<stdin>", line 1, in <module>
  File "/home/coder/third-party/polars/py-polars/polars/lazyframe/frame.py", line 2035, in collect
    return wrap_df(ldf.collect(callback))
polars.exceptions.ComputeError: 'cuda' conversion failed: NotImplementedError: ('Query execution with GPU not possible: unsupported operations.\nThe errors were:\n- NotImplementedError: anonymousfunction', [NotImplementedError('anonymousfunction')])
```

Currently, only the proximal cause of failure to execute on the GPU is reported, we plan to extend
this functionality to report all unsupported operations for a query.

### Testing

The Polars and NVIDIA RAPIDS teams run comprehensive unit and integration tests to ensure that the
GPU-accelerated Polars backend works smoothly.

The **full** Polars test suite is run on every commit made to the GPU engine, ensuring consistency
of results.

The GPU engine currently passes 99.2% of the Polars unit tests with CPU fallback enabled. Without
CPU fallback, the GPU engine passes 88.8% of the Polars unit tests. With fallback, there are
approximately 100 failing tests: around 40 of these fail due to mismatching debug output; there are
some cases where the GPU engine produces a correct result but uses a different data type; the
remainder are cases where we do not correctly determine that a query is unsupported and therefore
fail at runtime, instead of falling back.

### When Should I Use a GPU?

Based on our benchmarking, you're most likely to observe speedups using the GPU engine when your
workflow's profile is dominated by grouped aggregations and joins. In contrast I/O bound queries
typically show similar performance on GPU and CPU. Based on our testing, raw datasets of 1 TiB fit
(depending on the workflow) well with a GPU with 80GiB of memory.

### Providing feedback

Please report issues, and missing features, on the Polars
[issue tracker](https://github.com/pola-rs/polars/issues).
