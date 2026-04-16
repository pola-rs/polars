# Slurm

The quickstart below shows one way to run Polars queries on a Polars on-premises cluster using the
Slurm Workflow Manager. As this is not an introduction to Slurm and its intricacies we will focus on
making a barebone setup functional.

<!-- dprint-ignore-start -->

!!! info "Docker Compose setup"

    This how-to has been tested using the containerized setup from [this repo](https://github.com/giovtorres/slurm-docker-cluster.git).
    It has been judged to be representative enough of a bare, minimal Slurm setup.
    The only expected "extra" is a shared filesystem to exchange files between nodes (represented here by a volume mounted on all containers).

<!-- dprint-ignore-end -->

Since we only support static leader election for bare-metal, the information (name and address) of
the scheduler need to be known to all worker nodes _at startup_. In a real Slurm cluster each node
will have static characteristics, hostname and IP notably. These values are usually known
cluster-wide via a mapping stored on the filesystem, and a quick comparison with the
`SLURM_JOB_NODELIST` environment variable available upon submitting a job should be enough to infer
the IP of the service. A similar result could be obtained via the `SINFO` command.

In the current example we do not expect such a mapping to exist nor an unrestrained access to
`SINFO`, and we will fetch and store the information about the node hosting the scheduler on a
shared volume. The file is located at `/opt/polars/scheduler.txt`. The same volume will be used to
store the `polars-on-premises` binary as well as its associated `license.json` file. The `uv`
command necessary to set up the Python virtual environment is assumed to be available system-wide.

For simplicity we keep our cluster to two nodes (`#SBATCH --nodes=2`), and a single batch job
running two services (`#SBATCH --ntasks=2`). At the bottom of this page are reproduced the scripts
in their integrality.

## Cluster setup

### Scheduler

As mentioned above, the information about the scheduler need to be known to other nodes; as a first
step we write them to an accessible location (shared volume). We then generate the configuration
before creating the Python virtual environment and starting the service.

```bash
CLIENT_VERSION="0.6.0"
POLARS_VERSION="1.39.2"
PYTHON_VERSION="3.12"

POLARS_ONPREM_BIN="/opt/polars/polars-on-premises-0.2.3"

# Fetch and store scheduler node information
SCHEDULER_ID=$(hostname)
SCHEDULER_IP=$(hostname -I | awk '{print$1}')
echo -e "SCHEDULER_ID=$SCHEDULER_ID\nSCHEDULER_IP=$SCHEDULER_IP" > /opt/polars/scheduler.txt

# Generate the configuration for a scheduler + worker node
cat << EOF > /tmp/polars/config.toml
cluster_id = "polars-on-premises-cluster"
instance_id = "$SCHEDULER_ID"
license = "/opt/polars/polars-on-premises_license.json"

[scheduler]
enabled = true
allow_local_sinks = true
anonymous_result_location.local.path = "/tmp/polars/results-data"
n_workers = 2

[worker]
enabled = true
shuffle_location.local.path = "/tmp/polars/shuffle-data-path"
task_service.public_addr = "$SCHEDULER_IP"
shuffle_service.public_addr = "$SCHEDULER_IP"

[observatory]
enabled = true
max_metrics_bytes_total = 30000
database_path = "/tmp/polars/observatory/"

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "$SCHEDULER_ID"
observatory_service.public_addr = "$SCHEDULER_IP"
scheduler_service.public_addr = "$SCHEDULER_IP"
EOF

mkdir -p /tmp/polars/{observatory,results-data,shuffle-data-path}

# Create and activate the Python virtual environment
uv venv --allow-existing --managed-python --python=$PYTHON_VERSION
source .venv/bin/activate
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
uv add polars[cloudpickle]==$POLARS_VERSION polars-cloud==$CLIENT_VERSION

# Start the service
$POLARS_ONPREM_BIN service --config-path=/tmp/polars/config.toml &
```

<!-- dprint-ignore-start -->

!!! info "Observatory & query profiler"

    Depending on whether your Slurm cluster allows exposing a frontend, the Polars observatory and query profiler might become available for you to browse.
    The HTTP address will be set to the scheduler node, and the port to `3001` (default).
    Keep it mind that given the setup described here it will only remain available for the duration of the Slurm job.

<!-- dprint-ignore-end -->

### Worker(s)

The steps to deploy a worker are fairly identical, given the scheduler node information are
available. A slightly different configuration is generated (skipping a few unnecessary sections in
the TOML file).

```bash
CLIENT_VERSION="0.6.0"
POLARS_VERSION="1.39.2"
PYTHON_VERSION="3.12"

POLARS_ONPREM_BIN="/opt/polars/polars-on-premises-0.2.3"

# Fetch the scheduler node information
source /opt/polars/scheduler.txt
export $SCHEDULER_ID
export $SCHEDULER_IP

# Generate the configuration for a worker node
cat << EOF > /tmp/polars/config.toml
cluster_id = "polars-on-premises-cluster"
instance_id = "$(hostname)"
license = "/opt/polars/polars-on-premises_license.json"

[worker]
enabled = true
shuffle_location.local.path = "/tmp/polars/shuffle-data-path"
task_service.public_addr = "$SCHEDULER_IP"
shuffle_service.public_addr = "$SCHEDULER_IP"

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "$SCHEDULER_ID"
observatory_service.public_addr = "$SCHEDULER_IP"
scheduler_service.public_addr = "$SCHEDULER_IP"
EOF

mkdir -p /tmp/polars/{observatory,results-data,shuffle-data-path}

# Create and activate the Python virtual environment
uv venv --allow-existing --managed-python --python=$PYTHON_VERSION
source .venv/bin/activate
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')
uv add polars[cloudpickle]==$POLARS_VERSION polars-cloud==$CLIENT_VERSION

# Start the service
$POLARS_ONPREM_BIN service --config-path=/tmp/polars/config.toml &
```

### Submitting Polars queries

Once the cluster has been set up via the steps detailed above, Polars queries can be submitted to
the scheduler. Here too the node information are necessary, and need to be exposed; below we use the
same environment variable mechanism as earlier.

```bash
CLIENT_VERSION="0.6.0"
POLARS_VERSION="1.39.2"
PYTHON_VERSION="3.12"

# Make the scheduler node information known
source /opt/polars/scheduler.txt
export $SCHEDULER_IP

# Create and activate the Python virtual environment
uv venv --allow-existing --managed-python --python=$PYTHON_VERSION
source .venv/bin/activate
uv add polars[cloudpickle]==$POLARS_VERSION polars-cloud==$CLIENT_VERSION

# Submit query
python query.py
```

In the case one would submit several queries to be executed on the same cluster the snippet would
look as follows:

```bash
CLIENT_VERSION="0.6.0"
POLARS_VERSION="1.39.2"
PYTHON_VERSION="3.12"

# Make the scheduler node information known
source /opt/polars/scheduler.txt
export $SCHEDULER_IP

# Create and activate the Python virtual environment
uv venv --allow-existing --managed-python --python=$PYTHON_VERSION
source .venv/bin/activate
uv add polars[cloudpickle]==$POLARS_VERSION polars-cloud==$CLIENT_VERSION

# Submit queries
PID_QUERIES=()

python query1.py & PID_QUERIES+=($!)  # Store query PID
python query2.py & PID_QUERIES+=($!)
python query3.py & PID_QUERIES+=($!)

# Wait for all queries to finish, set job exit code to 1 if any failed
EXIT_CODE=0
for pid in "${PID_QUERIES[@]}"; do
  wait $pid || EXIT_CODE=1  # Store query exit code
done
exit $EXIT_CODE
```

For reference, a Polars query structure should be similar to:

```python
import os
import datetime

import polars as pl
import polars_cloud as pc

ctx = pc.ClusterContext(compute_address=os.environ["SCHEDULER_IP"])

(
    pl.scan_parquet(...)
    .filter(pl.col("date_column") <= datetime.date(2025, 9, 2))
    .group_by("column_a", "column_b")
    .remote(ctx)
    .execute()
    .sink_parquet(...)
)
```

## Full setup

As you may have noticed quite a lot of the code presented above overlap and could be factored in its
own script or set of functions. The configuration files could also be templated as plain files and
rendered on the fly via a `sed` command for instance.

Below a full example, split in two scripts: the main job (`submit.sh`) submitted via `sbatch`,
calling a helper (`setup-service.sh`) via `srun` where necessary.

In case the Python virtual environment is not shared amongst Slurm nodes and set up in the
`setup-service.sh` helper script instead, a `sleep` statement might be required after each `srun`
call to leave some time for `uv` to finalize before starting the respective services.

- `submit.sh`: cluster setup and query submission

```bash
#!/bin/bash
#SBATCH --nodes=2
#SBATCH --ntasks=2

INFO_FILE="/opt/polars/scheduler.txt"
VENV_PATH="/opt/polars/venv"

# Create a shared Python virtual environment
uv venv --allow-existing --managed-python --python=$PYTHON_VERSION $VENV_PATH
source $VENV_PATH/bin/activate
uv add polars[cloudpickle]==$POLARS_VERSION polars-cloud==$CLIENT_VERSION

# Start scheduler + worker
SERVICE=scheduler srun setup-service.sh &

# Start worker(s)
SERVICE=worker srun setup-service.sh &

# Fetch the scheduler node information
[ -f $INFO_FILE ] || (echo "Error: info file not found at $INFO_FILE"; exit 1)
source $INFO_FILE
export $SCHEDULER_IP

# Submit queries
PID_QUERIES=()

python query1.py & PID_QUERIES+=($!)
python query2.py & PID_QUERIES+=($!)

EXIT_CODE=0
for pid in "${PID_QUERIES[@]}"; do
  wait $pid || EXIT_CODE=1
done
exit $EXIT_CODE
```

- `setup-service.sh`: service setup

```bash
#!/bin/bash

INFO_FILE="/opt/polars/scheduler.txt"
VENV_PATH="/opt/polars/venv"

CLIENT_VERSION="0.6.0"
POLARS_VERSION="1.39.2"
PYTHON_VERSION="3.12"

ANON_RESULTS_PATH="/tmp/polars/results-data"
OBSERVATORY_DB_PATH="/tmp/polars/observatory"
SHUFFLE_DATA_PATH="/tmp/polars/shuffle-data-path"

POLARS_ONPREM_BIN="/opt/polars/polars-on-premises-0.2.3"

if [ "$SERVICE" = "scheduler" ]; then

  # Fetch and store scheduler node information
  SCHEDULER_ID=$(hostname)
  SCHEDULER_IP=$(hostname -I | awk '{print$1}')
  echo -e "SCHEDULER_ID=$SCHEDULER_ID\nSCHEDULER_IP=$SCHEDULER_IP" > $INFO_FILE

  # Generate the configuration for a scheduler + worker node
  cat config-scheduler.tpl | env \
    scheduler_id=$SCHEDULER_ID \
    scheduler_ip=$SCHEDULER_IP \
    anon_results_path=$ANON_RESULTS_PATH \
    observatory_db_path=$OBSERVATORY_DB_PATH \
    shuffle_data_path=$SHUFFLE_DATA_PATH \
    envsubst > /tmp/polars/config.toml

elif [ "$SERVICE" = "worker" ]; then
  
  # Fetch the scheduler node information
  [ -f $INFO_FILE ] || (echo "Error: info file not found at $INFO_FILE"; exit 1)
  source $INFO_FILE
  export $SCHEDULER_ID
  export $SCHEDULER_IP

  # Generate the configuration for a worker node
  cat config-worker.tpl | env \
    scheduler_id=$SCHEDULER_ID \
    scheduler_ip=$SCHEDULER_IP \
    shuffle_data_path=$SHUFFLE_DATA_PATH \
    envsubst > /tmp/polars/config.toml

fi

mkdir -p $ANON_RESULTS_PATH $OBSERVATORY_DB_PATH $SHUFFLE_DATA_PATH

# Activate the shared Python virtual environment
source $VENV_PATH/bin/activate
export LD_LIBRARY_PATH=$(python -c 'import sysconfig; print(sysconfig.get_config_var("LIBDIR"))')

# Start the service
$POLARS_ONPREM_BIN service --config-path=/tmp/polars/config.toml
```
