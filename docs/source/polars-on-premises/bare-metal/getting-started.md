First of all, make sure to obtain a license for Polars on-premises by
[signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). You will receive a JSON-formatted
license key with which you can download Polars On-Premises.

## Downloading Polars On-Premises

#### Using UV

The simplest way to get started is to use our private PyPi index. You can log in to the index using
the given license key. This will automatically install the correct Polars version and work well
within virtual environments.

```shell
$ export LICENSE_KEY=$(cat license.json)
$ uv auth login https://get.onprem.pola.rs/pypi/simple --token $LICENSE_KEY
$ uv venv && source .venv/bin/activate
$ uv pip install --index-url=https://get.onprem.pola.rs/pypi/simple polars-on-premises==0.1.0
```

The `polars-on-premises` command will then be available within your virtual environment.

#### Downloading the server binary only

You may also download the binary directly using the curl command below. Note that this still
requires your system to have a Python interpreter available, and Polars to be installed. There are
also some

```shell
$ export LICENSE_KEY=$(cat license.json)
$ curl -L 'https://get.onprem.pola.rs?version=0.1.0' --data $LICENSE_KEY --output polars-on-premises
```

See the [Python Environments page](/polars-on-premises/bare-metal/python-environment) for more
information about setting up Polars On-Premises in your environment.

## Reading the license

The license can be read by running the following command:

```shell
$ polars-on-premises service --print-eula /path/to/license.json
```

## Running the binary

The main entrypoint is as follows:

```shell
$ polars-on-premises service --config-path /etc/polars-cloud/config.toml
```

However, the service requires quite some configuration to get started. Below you can find an example
scheduler and worker config, and you can find the full configuration reference
[here](/polars-on-premises/bare-metal/config-reference).

## Quick start

To get started fast, you can use the following configuration. It enables the scheduler, worker,
observatory, and monitoring components. It writes query output data and shuffle data to a local
directory.

```toml
cluster_id = "polars-cluster"
instance_id = "node-0"
license = "./license.json" # Path to your Polars on-premises license. This is a JSON file containing your company name, license expiry, and license signature.

# Component that receives the Polars queries from the Python client.
[scheduler]
enabled = true
allow_local_sinks = true
anonymous_result_location.local.path = "./results-data"
n_workers = 1

# Component that receives and executes tasks from the scheduler.
[worker]
enabled = true
shuffle_location.local.path = "./shuffle-data-path"
task_service.public_addr = "127.0.0.1"
shuffle_service.public_addr = "127.0.0.1"

# Component that receives query profiling and host metrics.
[observatory]
enabled = true
max_metrics_bytes_total = 30000
database_path = "./observatory/"

# Enables exporting query profiles and host metrics to the observatory service.
[monitoring]
enabled = true

# Explicitly define that node-0 is the leader node. The leader node should run the observatory and monitoring components.
[static_leader]
leader_instance_id = "node-0"
observatory_service.public_addr = "127.0.0.1"
scheduler_service.public_addr = "127.0.0.1"
```

## Configuration

The complete configuration reference can be found
[here](/polars-on-premises/bare-metal/config-reference).
