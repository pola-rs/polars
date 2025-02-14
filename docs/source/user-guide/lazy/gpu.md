# GPU Support

Polars provides an in-memory, GPU-accelerated execution engine for the Lazy API in Python using
[RAPIDS cuDF](https://docs.rapids.ai/api/cudf/stable/) on NVIDIA GPUs. This functionality is
available in Open Beta, is undergoing rapid development, and is currently a single GPU
implementation.

If you install Polars with the [GPU feature flag](../installation.md), you can trigger GPU-based
execution by running `.collect(engine="gpu")` instead of `.collect()`.

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

Learn more in the [GPU Support guide](../gpu-support.md).
