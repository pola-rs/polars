# Resource limits

Polars will always attempt to use all available CPU resources, and consumes memory resources as
needed. If possible, assign physical cores to Polars to avoid contention with other processes.
Polars on-premises workers consist of a main process and an executor process, with the latter doing
most computation. If the executor dies, it will lose all progress of the stage it was working on.
All progress of previous stages (i.e. shuffle data) is managed by the main process.

In other words, if the system is low on memory, the first process that should be killed, is the
executor process. Polars on-premises will already automatically configure `oom-score-adj` on its
executor process.

If there are other system critical processes, we recommend either delegating a cgroup to Polars
on-premises, or manually setting up cgroup limits for the entire Polars on-premises service.

### Delegating cgroup to Polars on-premises

Cgroups can contain subgroups, each with independent limits. Polars on-premises can create these
cgroups, and choose proper memory limits for each of its components. To use this feature, ensure you
delegate cgroups to the Polars On-Premise process and configure `memory_limit` in the configuration
file.

```toml
cluster_id = "polars-cluster"
instance_id = "node-0"
license = "/etc/polars/license.json"
memory_limit = 10737418240 # 10 GiB
# ...
```

### Manually configuring cgroup limits

You can also manually configure a memory limit on a cgroup containing all the processes. For example
using Systemd's `resource-control`. The disadvantage of this approach is that the individual
components will contend for the same memory capacity, which may prevent Polars on-premises from
gracefully handling OOM errors on the executor.
