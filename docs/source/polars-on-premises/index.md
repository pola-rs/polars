# Polars On-Prem

Polars On-Prem lets you host a Polars cluster on your own hardware or cloud provider. Once
connected, you can run queries remotely, either executing multiple queries in parallel, or
distributing a single query across multiple nodes for large-scale workloads.

Interested in running Polars On-Prem?
[Sign up here to apply](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv).

After installing Polars On-Prem either on bare-metal or on Kubernetes, you can connect to your
cluster using the Polars Cloud Python client.

```python
import polars as pl
import polars_cloud as pc

# Connect to your Polars On-Prem cluster
ctx = pc.ClusterContext(compute_address="your-cluster-compute-address")
result = (
    pl.LazyFrame()
    .with_columns(a=pl.arange(0, 100000000).sum())
    .remote(ctx)
    .distributed()
    .execute()
)
print(result)
```
