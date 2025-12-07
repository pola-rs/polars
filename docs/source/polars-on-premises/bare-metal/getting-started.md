First of all, make sure to obtain a license for Polars on-premises by
[signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). You will receive a link to download
our binary named `polars-on-premises` as well as a license for running Polars on-premises.

## Running the binary

The main entrypoint is as follows:

```shell
$ polars-on-premises service --config-path /etc/polars-cloud/config.toml
```

However, the service requires quite some configuration to get started. Below you can find an example
scheduler and worker config, and you can find the full configuration reference
[here](/polars-on-premises/bare-metal/config-reference).

## Example scheduler config

Here is a cleaned-up example you can use after the reference tables. It keeps the scheduler
single-purpose (no worker role) and turns on observability.

```toml
cluster_id = "foobarbaz"
cublet_id = "scheduler"
license = "/etc/polars/license.json"
memory_limit = 1073741824 # 1 GiB

[scheduler]
enabled = true
anonymous_result_dst = "s3://my-bucket/path/to/dir"
allow_shared_disk = false
n_workers = 4

[observatory]
enabled = true
max_metrics_bytes_total = 0

[static_leader]
leader_key = "scheduler"
public_leader_addr = "192.168.1.1"

[service]
public_address = "192.168.1.1"
auth = "insecure"
connection = "insecure"
```

## Example worker config

And the matching worker config. This example gives the worker a local shuffle path and enables
observability.

```toml
cluster_id = "foobarbaz"
cublet_id = "worker_0"
license = "/etc/polars/license.json"
memory_limit = 10737418240 # 10 GiB

[worker]
enabled = true
worker_ip = "192.168.1.2"
flight_port = 5052
service_port = 5053
heartbeat_interval_secs = 5
shuffle_data_path = "/opt/shuffle-data-path"

[observatory]
enabled = true
max_metrics_bytes_total = 0

[static_leader]
leader_key = "scheduler"
public_leader_addr = "192.168.1.1"
```

The complete configuration reference can be found
[here](/polars-on-premises/bare-metal/config-reference).
