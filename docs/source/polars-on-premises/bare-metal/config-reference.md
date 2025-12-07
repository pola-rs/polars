# Config file reference

This page describes the different configuration options for polars-on-premises. The config file is a
standard TOML file with different sections. Any of the configuration can be overridden using
environment variables in the following format: `PC_CUBLET__section_name__key`.

### Top-level configuration

| Key            | Type            | Description                                                                                                                                                   |
| -------------- | --------------- | ------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `cluster_id`   | string          | Logical ID for the cluster; workers and scheduler that share this ID will form a single cluster.<br>e.g. `prod-eu-1`; must be unique among all clusters.      |
| `cublet_id`    | string          | Unique ID for this node ("cublet") within the cluster, used for addressing and leader selection.<br>e.g. `scheduler`, `worker_0`; must be unique per cluster. |
| `license`      | path            | Absolute path to the polars-on-premises license file required to start the process.<br>e.g. `/etc/polars/license.json`.                                       |
| `memory_limit` | integer (bytes) | Hard memory budget for all components in this cublet; enforced via cgroups when delegated.<br>e.g. `1073741824` (1 GiB), `10737418240` (10 GiB).              |

### `[scheduler]` section

| Key                    | Type         | Description                                                                                                                                                                                                 |
| ---------------------- | ------------ | ----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`              | bool         | Whether the scheduler component runs in this process.<br> `true` for the leader node, `false` on pure workers.                                                                                              |
| `anonymous_result_dst` | string (URI) | Destination for results of queries that don’t have an explicit sink. Currently S3-only storage supported. Bucket must be reachable by scheduler, workers, and clients.<br>e.g. `s3://my-bucket/path/to/dir` |
| `allow_shared_disk`    | bool         | Whether workers are allowed to write to a shared/local disk visible to the scheduler.<br> `false` for fully remote/storage-only setups; `true` if you have a shared filesystem.                             |
| `n_workers`            | int          | Expected number of workers in this cluster; scheduler waits for this many to be online before running queries.<br>e.g. `4`                                                                                  |

### `[worker]` section

| Key                       | Type   | Description                                                                                                             |
| ------------------------- | ------ | ----------------------------------------------------------------------------------------------------------------------- |
| `enabled`                 | bool   | Whether the worker component runs in this process.<br> `true` on worker nodes, `false` on the dedicated scheduler.      |
| `worker_ip`               | string | Public or routable IP address other workers/scheduler use to reach this worker.<br>e.g. `192.168.1.2`                   |
| `flight_port`             | int    | Port for shuffle traffic between workers.<br>e.g. `5052`                                                                |
| `service_port`            | int    | Port on which the worker receives task instructions from the scheduler.<br>e.g. `5053`                                  |
| `heartbeat_interval_secs` | int    | Interval for worker heartbeats towards the scheduler, used for liveness and load reporting.<br>e.g. `5`                 |
| `shuffle_data_path`       | path   | Local path where shuffle / intermediate data is stored; fast local SSD is recommended.<br>e.g. `/opt/shuffle-data-path` |

### `[observatory]` section

| Key                       | Type | Description                                                                                                                                                                                                                                                           |
| ------------------------- | ---- | --------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------- |
| `enabled`                 | bool | Enable sending/receiving profiling data so clients can call `result.await_profile()`.<br> `true` on both scheduler and workers if you want profiles on queries; `false` to disable.                                                                                   |
| `max_metrics_bytes_total` | int  | How many bytes all the worker host metrics will consume in total. If a system-wide memory limit is specified then this is added to the share that the scheduler takes. Note that the worker host metrics is not yet available, so this configuration can be set to 0. |

### `[static_leader]` section

| Key                  | Type   | Description                                                                                                               |
| -------------------- | ------ | ------------------------------------------------------------------------------------------------------------------------- |
| `leader_key`         | string | ID of the leader cublet; should match the scheduler’s `cublet_id`.<br>Typically `scheduler` to match your scheduler node. |
| `public_leader_addr` | string | Host/IP where the leader’s `[service]` is reachable from this node.<br>e.g. `192.168.1.1`                                 |

### `[service]` section

| Key              | Type   | Description                                                                                                               |
| ---------------- | ------ | ------------------------------------------------------------------------------------------------------------------------- |
| `public_address` | string | ID of the leader cublet; should match the scheduler’s `cublet_id`.<br>Typically `scheduler` to match your scheduler node. |
| `auth`           | string | Host/IP where the leader’s `[service]` is reachable from this node.<br>e.g. `192.168.1.1`                                 |
| `connection`     | string | Host/IP where the leader’s `[service]` is reachable from this node.<br>e.g. `192.168.1.1`                                 |
