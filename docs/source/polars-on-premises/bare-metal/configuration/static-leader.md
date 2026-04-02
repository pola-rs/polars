# Static leader configuration

All nodes in the cluster need to know whether they are the leader or if not, where to find the
leader. Generally, this section can be identical across all nodes. In this section, we need to
specify the instance id of the leader node, and the public addresses of the observatory and
scheduler services.

```toml
[static_leader]
leader_instance_id = "node-1"
observatory_service.public_addr = "127.0.0.1:4003"
scheduler_service.public_addr = "127.0.0.1:4002"
```

If the current node's instance id matches the `leader_instance_id`, it will run the observatory and
scheduler services (if those components are enabled). This setup allows you to use a single
configuration file across all nodes in the cluster, only overriding the `instance_id` on each node.

You can set any configuration option in the file using an environment variable in the format of
`PC_CUBLET__<SECTION>__<KEY>`. For example, to override the `instance_id` for a node, you can set
the environment variable `PC_CUBLET__instance_id=node-2`, and to disable the host metrics collection
for a node, you can set the environment variable
`PC_CUBLET__observatory__host_metrics__enabled=false`.
