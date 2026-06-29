# Proxy mode

Proxy mode routes queries through the Polars Cloud control plane rather than direct cluster
connections. This approach provides enhanced security by eliminating the need to expose network
ports on your compute cluster.

![Architectural overview of proxy mode](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/proxy-mode.png)

In proxy mode the control plane polls the compute plane to retrieve the latest information about
running compute and queries.

## Reconnect to active clusters

Clusters that are started in `proxy` mode allow you and your team to reconnect to the instance while
it is active. There is more information available on the page about
[reconnecting to clusters](../context/reconnect.md).
