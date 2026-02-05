# Example configurations

### Ports updates

<details>

<summary>Cluster with all ports updated (single node)</summary>

```toml
cluster_id = "cluster-1"
instance_id = "node-1"
license = "/etc/polars/license.json"

[scheduler]
enabled = true
n_workers = 1
client_service.bind_addr.port = 4001
worker_service.bind_addr.port = 4002

[worker]
enabled = true
task_service.bind_addr.port = 4005
task_service.public_addr = "127.0.0.1:4005"
shuffle_service.bind_addr.port = 4006
shuffle_service.public_addr = "127.0.0.1:4006"

[observatory]
enabled = true
max_metrics_bytes_total = 180000
database_path = "/opt/db/observatory.db"
service.bind_addr.port = 4003

[observatory.rest_api]
enabled = true
service.bind_addr.port = 4004

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "node-1"
observatory_service.public_addr = "127.0.0.1:4003"
scheduler_service.public_addr = "127.0.0.1:4002"
```

</details>

<details>

<summary>Cluster with all ports updated (multi node)</summary>

Scheduler

```toml
cluster_id = "cluster-1"
instance_id = "node-1"

[scheduler]
enabled = true
n_workers = 1
client_service.bind_addr.port = 4001
worker_service.bind_addr.port = 4002

[observatory]
enabled = true
max_metrics_bytes_total = 180000
database_path = "/opt/db/observatory.db"
service.bind_addr.port = 4003

[observatory.rest_api]
enabled = true
service.bind_addr.port = 4004

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "node-1"
observatory_service.public_addr = "127.0.0.1:4003"
scheduler_service.public_addr = "127.0.0.1:4002"
```

Worker

```toml
cluster_id = "cluster-1"
instance_id = "node-2"

[worker]
enabled = true
task_service.bind_addr.port = 4005
task_service.public_addr = "127.0.0.1:4005"
shuffle_service.bind_addr.port = 4006
shuffle_service.public_addr = "127.0.0.1:4006"

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "node-1"
observatory_service.public_addr = "127.0.0.1:4003"
scheduler_service.public_addr = "127.0.0.1:4002"
```

</details>
