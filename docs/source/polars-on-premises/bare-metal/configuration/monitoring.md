# Profiling and host metrics

The profiling and host metrics system consists of a sender and receiving side. The observatory
component's main task is to receive telemetry data from the cluster's nodes. The observatory
component can currently only be enabled on nodes with the scheduler component enabled. The
monitoring component is used to send telemetry data to the observatory. An example configuration is
shown below.

```toml
# ... global configuration

[scheduler]
enabled = true
# ... continue configuration for scheduler

[observatory]
enabled = true
max_metrics_bytes_total = 5760000
database_path = "./observatory/"

[monitoring]
enabled = true
```

## Observatory data stores

The observatory stores profiling data in an SQLite database file. The location of this file can be
configured as shown above using the `database_path` key. The storage needed for this file is in the
order of several tens of MB, depending on the number of queries and their complexity.

The observatory stores host metrics in a pre-allocated buffer in memory. The size of this buffer can
be configured using the `max_metrics_bytes_total` key. And since the required memory for these
metrics can quickly rise if a cluster is kept alive for a long time, there is no default value for
this key. You can estimate the required memory at around 50 bytes per second per worker node. So for
a cluster with 32 worker nodes, with a history of 1 hour, you would need around
`50 * 32 * 3600 = 5760000` bytes, or 5.7 MB.

## Host metrics exporter

The host metrics exporter supports collecting CPU and memory usage metrics from the process tree,
and cgroups v1 and v2. Since traversing the process tree, and collecting metrics for each process is
quite compute expensive, and a cgroup's usage can be instantly queried, we recommend to always run
Polars on-premises in a cgroup.

If you don't use the dashboard's host metrics feature, we recommend disabling the host metrics
exporter.

```toml
[monitoring]
enabled = true

[monitoring.host_metrics]
enabled = false
```

## Disabling observatory and monitoring

If you do not intend to use the dashboard, or the Python client's `result.await_profile()` method,
you can disable the observatory and monitoring components completely. Queries will still be
executed, but profiling data will not be collected.

```toml
[observatory]
enabled = false

[monitoring]
enabled = false
```
