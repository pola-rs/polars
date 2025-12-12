# Config file reference

This page describes the different configuration options for Polars on-premises.
The config file is a standard TOML file with different sections.
Any of the configuration can be overridden using environment variables in the following format: `PC_CUBLET__section_name__key`.

### Top-level configuration

The `polars-on-premises` binary requires a license which path is provided as a configuration option, listed below.
The license itself has the following shape:

```json
{"params":{"expiry":"2026-01-31T23:59:59Z","name":"Company"},"signature":"..."}
```

| Key            | Type    | Description                                                                                                                                                         |
| -------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cluster_id`   | string  | Logical ID for the cluster; workers and scheduler that share this ID will form a single cluster.<br>e.g. `prod-eu-1`; must be unique among all clusters.            |
| `cublet_id`    | string  | Unique ID for this node (_aka_ "cublet") within the cluster, used for addressing and leader selection.<br>e.g. `scheduler`, `worker_0`; must be unique per cluster. |
| `license`      | path    | Absolute path to the Polars on-premises license file required to start the process.<br>e.g. `/etc/polars/license.json`.                                             |
| `memory_limit` | integer | Hard memory budget for all components in this cublet; enforced via cgroups when delegated.<br>e.g. `1073741824` (1 GiB), `10737418240` (10 GiB).                    |

Example:

```toml
cluster_id = "polars-cluster-dev"
cublet_id = "scheduler"
license = "/etc/polars/license.json"
memory_limit = 1073741824 # 1 GiB
```

### `[scheduler]` section

For remote Polars queries without a specific output sink, Polars on-premises can automatically add persistent sink.
We call these sinks "anonymous results" sinks.
Infrastructure-wise, these sinks are backed by S3-compatible storage accessible from all worker nodes and the Python client.
The data written to this location is not automatically deleted, so you need to configure a retention policy for this data yourself.

You may configure the credentials using the options listed below; the key names correspond to the [`storage_options` parameter from the `scan_parquet()` method](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html) (_e.g._ `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `aws_region`).
We currently only support the AWS keys of the `storage_options` dictionary, but note that you can use any other cloud provider that supports the S3 API, such as MinIO or DigitalOcean Spaces.

| Key                                             | Type    | Description                                                                                                                                                                                                                   |
| ----------------------------------------------- | ------- | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                                       | boolean | Whether the scheduler component runs in this process.<br>`true` for the leader node, `false` on pure workers.                                                                                                                 |
| `allow_shared_disk`                             | boolean | Whether workers are allowed to write to a shared/local disk visible to the scheduler.<br>`false` for fully remote/storage-only setups, `true` if you have a shared filesystem.                                                |
| `n_workers`                                     | integer | Expected number of workers in this cluster; scheduler waits for the latter to be online before running queries.<br>e.g. `4`.                                                                                                  |
| `anonymous_result_dst.s3.url`                   | string  | Destination for results of queries that do not have an explicit sink. Currently only S3-compatible storages are supported. The bucket must be reachable by scheduler, workers, and client.<br>e.g. `s3://bucket/path/to/key`. |
| `anonymous_result_dst.s3.aws_endpoint_url`      | string  | Storage option configuration, see [`scan_parquet()`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html).                                                                                          |
| `anonymous_result_dst.s3.aws_region`            | string  | Storage option configuration.<br/>e.g. `eu-east-1`                                                                                                                                                                            |
| `anonymous_result_dst.s3.aws_access_key_id`     | string  | Storage option configuration.                                                                                                                                                                                                 |
| `anonymous_result_dst.s3.aws_secret_access_key` | string  | Storage option configuration.                                                                                                                                                                                                 |

Example:

```toml
[scheduler]
enabled = true
allow_shared_disk = false
n_workers = 4
anonymous_result_dst.s3.url = "s3://bucket/path/to/key"
```

### `[worker]` section

During distributed query execution, data may be shuffled between workers.
A local path can be provided, but shuffles can also be configured to use S3-compatible storage (accessible from all worker nodes).
You may configure the credentials using the options listed below; the key names correspond to the [`storage_options` parameter from the `scan_parquet()` method](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html) (_e.g._ `aws_access_key_id`, `aws_secret_access_key`, `aws_session_token`, `aws_region`).

| Key                                         | Type    | Description                                                                                                                          |
| ------------------------------------------- | ------- | ------------------------------------------------------------------------------------------------------------------------------------ |
| `enabled`                                   | boolean | Whether the worker component runs in this process.<br>`true` on worker nodes, `false` on the dedicated scheduler.                    |
| `worker_ip`                                 | string  | Public or routable IP address other workers/scheduler use to reach this worker.<br>e.g. `192.168.1.2`.                               |
| `flight_port`                               | integer | Port for shuffle traffic between workers.<br>e.g. `5052`.                                                                            |
| `service_port`                              | integer | Port on which the worker receives task instructions from the scheduler.<br>e.g. `5053`.                                              |
| `heartbeat_interval_secs`                   | integer | Interval for worker heartbeats towards the scheduler, used for liveness and load reporting.<br>e.g. `5`.                             |
| `shuffle_location.local.path`               | path    | Local path where shuffle/intermediate data is stored; fast local SSD is recommended.<br>e.g. `/mnt/storage/polars/shuffle`.          |
| `shuffle_location.s3.url`                   | path    | Destination for shuffle/intermediate data.<br>e.g. `s3://bucket/path/to/key`.                                                        |
| `shuffle_location.s3.aws_endpoint_url`      | string  | Storage option configuration, see [`scan_parquet()`](https://docs.pola.rs/api/python/stable/reference/api/polars.scan_parquet.html). |
| `shuffle_location.s3.aws_region`            | string  | Storage option configuration.<br/>e.g. `eu-east-1`                                                                                   |
| `shuffle_location.s3.aws_access_key_id`     | string  | Storage option configuration.                                                                                                        |
| `shuffle_location.s3.aws_secret_access_key` | string  | Storage option configuration.                                                                                                        |

Example:

```toml
[worker]
enabled = true
worker_ip = "192.168.1.2"
flight_port = 5052
service_port = 5053
heartbeat_interval_secs = 5
shuffle_location.local.path = "/mnt/storage/polars/shuffle"
```

### `[observatory]` section

| Key                       | Type    | Description                                                                                                                                                                                                                                                           |
| ------------------------- | ------- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                 | boolean | Enable sending/receiving profiling data so clients can call `result.await_profile()`.<br>`true` on both scheduler and workers if you want profiles on queries; `false` to disable.                                                                                    |
| `max_metrics_bytes_total` | integer | How many bytes all the worker host metrics will consume in total. If a system-wide memory limit is specified then this is added to the share that the scheduler takes. Note that the worker host metrics is not yet available, so this configuration can be set to 0. |

Example:

```toml
[observatory]
enabled = true
max_metrics_bytes_total = 0
```

### `[static_leader]` section

| Key                  | Type   | Description                                                                                                                |
| -------------------- | ------ | -------------------------------------------------------------------------------------------------------------------------- |
| `leader_key`         | string | ID of the leader service; should match the scheduler’s `cublet_id`.<br>Typically `scheduler` to match your scheduler node. |
| `public_leader_addr` | string | Host/IP at which the leader is reachable from this node.<br>e.g. `192.168.1.1`.                                            |

Example:

```toml
[static_leader]
leader_key = "scheduler"
public_leader_addr = "192.168.1.1"
```
