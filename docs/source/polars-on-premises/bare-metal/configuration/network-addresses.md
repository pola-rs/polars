# Network addresses

Polars on-premises exposes several network services for communication between nodes. Each of these
services can be configured to bind to a specific network address. This can be important when running
a node on a specific network interface or when preventing port collisions. Most of the addresses are
configured by default, except:

- The worker services require a public address when the bind address is `0.0.0.0`.
- The `static_leader` section requires the public address of both the scheduler and observatory
  services.

## Scheduler

The scheduler has two main services: the client service and the worker service. The client service
is used by the Python client to submit queries and receive query status updates. The worker service
is used by the workers to register and send heartbeats. By default, they bind to `0.0.0.0:5051` and
`0.0.0.0:5050`, respectively.

```toml
instance_id = "scheduler"

[scheduler]
enabled = true
client_service.bind_addr = "0.0.0.0:5051"
worker_service.bind_addr = "0.0.0.0:5050"
# ...

[static_leader]
leader_instance_id = "scheduler"
scheduler_service.public_addr = "192.168.1.2:5051"
# ...
```

```toml
instance_id = "worker-0"

[worker]
enabled = true
# ...

[static_leader]
leader_instance_id = "scheduler"
scheduler_service.public_addr = "192.168.1.2:5051"
# ...
```

## Observatory

The observatory consists of two main components: the profiler and the dashboard with its REST API.
The profiler service receives OpenTelemetry traces from all cluster nodes. The other service exposes
an API for accessing the query plans, profiling data, and host metrics.

The observatory service (profiling) binds to `0.0.0.0:5049` by default. To enable sending traces to
the observatory, you need to enable the `monitoring` section in the configuration of the second
node, and configure the observatory service public address in the static leader section on the
worker nodes.

```toml
instance_id = "scheduler"
# ...

[scheduler]
enabled = true
# ...

[observatory]
enabled = true
service.bind_addr = "0.0.0.0:5049" # Defaults to 0.0.0.0:5049
[observatory.rest_api]
enabled = true
service.bind_addr = "0.0.0.0:3001" # Defaults to 0.0.0.0:3001

[static_leader]
leader_instance_id = "scheduler"
observatory_service.public_addr = "192.168.1.2:5049"
# ...
```

```toml
instance_id = "worker-0"
# ...

[worker]
enabled = true
# ...

[monitoring]
enabled = true

[static_leader]
leader_instance_id = "scheduler"
observatory_service.public_addr = "192.168.1.2:5049"
# ...
```

## Workers

The worker has two services, the task service and the shuffle service. The task service is used by
the scheduler to send tasks to the worker, and the shuffle service is used by the worker to serve
shuffles between each other. By default, they bind to `0.0.0.0:5052` and `0.0.0.0:5053`,
respectively. The shuffle service is only bound to when using local shuffle storage.

The worker services need to also have configured a public address. This address is used by the
scheduler to reach the worker services. If the bind address is `0.0.0.0`, the public address is
required. The public address can also be configured when the ports go through a NAT or load
balancer.

The following example shows how to change the bind and public addresses for the worker services:

```toml
[worker]
enabled = true
task_service.bind_addr.port = 4005
task_service.public_addr = "192.168.1.1:4005"
shuffle_service.bind_addr.port = 4006
shuffle_service.public_addr = "192.168.1.1:4006"
```
