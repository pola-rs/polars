# Exposing Polars on-premise

To use Polars on-premises from the Python client, the scheduler endpoint must be reachable from the
client. By default, the chart creates a `ClusterIP` service for the scheduler, which is only
reachable from within the cluster and through port-forwards:

```shell
kubectl port-forward svc/polars-scheduler 5051:5051
kubectl port-forward svc/polars-observatory 3001:3001
```

To expose the scheduler outside the cluster, you can change the `scheduler.services.scheduler.type`
value to `LoadBalancer` or `NodePort`. We recommend using the `LoadBalancer` type and configuring
TLS such that the connection to the cluster is encrypted. If you decide to use an insecure
connection, you must set `insecure=True` in the `ClusterContext`.

When the python client can't reach the scheduler, it will fail with a connection timeout like this:

```
RuntimeError: Error setting up gRPC connection, transport error
tcp connect error
Hint: you may need to restart the query if this error persists
```
