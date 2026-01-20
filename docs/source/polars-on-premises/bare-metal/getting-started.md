First of all, make sure to obtain a license for Polars on-premises by
[signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). You will receive a link to download
our binary named `polars-on-premises` as well as a JSON-formatted license for running Polars
on-premises.

## Reading the license

The license can be read by running the following command:

```shell
$ ./polars-on-premises service --print-eula /path/to/license.json
```

## Running the binary

The main entrypoint is as follows:

```shell
$ ./polars-on-premises service --config-path /etc/polars-cloud/config.toml
```

However, the service requires quite some configuration to get started. Below you can find an example
scheduler and worker config, and you can find the full configuration reference
[here](/polars-on-premises/bare-metal/config-reference).

## Configuration

The complete configuration reference can be found
[here](/polars-on-premises/bare-metal/config-reference).

### Example scheduler config

Here is a cleaned-up example you can use after the reference tables. It keeps the scheduler
single-purpose (no worker role) and turns on observability.

```toml
cluster_id = "polars-cluster"
instance_id = "scheduler"
license = "/etc/polars/license.json"
memory_limit = 1073741824 # 1 GiB

[scheduler]
enabled = true
allow_local_sinks = false
anonymous_result_location.s3.url = "s3://bucket/path/to/key"
n_workers = 4

[observatory]
enabled = true
max_metrics_bytes_total = 0

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "scheduler"
observatory_service.public_addr = "192.168.1.1"
scheduler_service.public_addr = "192.168.1.1"
```

### Example worker config

And the matching worker config. This example gives the worker a local shuffle path and enables
observability.

```toml
cluster_id = "polars-cluster"
instance_id = "worker_0"
license = "/etc/polars/license.json"
memory_limit = 10737418240 # 10 GiB

[worker]
enabled = true
shuffle_location.local.path = "/opt/shuffle-data-path"

[observatory]
enabled = true
max_metrics_bytes_total = 0

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "scheduler"
observatory_service.public_addr = "192.168.1.1"
scheduler_service.public_addr = "192.168.1.1"
```
