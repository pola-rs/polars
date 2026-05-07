# Resource allocation

# Resource allocation and node selectors

Most of the time, it is a good idea to run Polars On-Prem on dedicated nodes, with only one
worker pod per node.

First, figure out the available resources on your nodes. This is usually lower than the actual node
resources, as Kubernetes reserves some resources for system daemons. For example, if you have a
cluster of 3 `m4.xlarge` nodes (4 vCPUs, 16GiB memory), you may have 3770m CPU and 14.31GiB memory
available.

```yaml
worker:
  deployment:
    replicaCount: 3

    runtimeContainer:
      resources:
        requests:
          cpu: 3770m
          memory: 14.31GiB
   
        limits:
          cpu: 3770m
          memory: 14.31GiB
```

If your cluster has other workloads running on it, we still recommend running Polars On-Prem on
dedicated nodes, and using node selectors and taints/tolerations to ensure that only Polars
On-Prem pods are scheduled on those nodes. For example, you can add a node selector, toleration,
and affinity rules like this:

```yaml
worker:
  deployment:
    nodeSelector:
      kubernetes.io/e2e-az-name: e2e-az1
    tolerations:
    - key: "key"
      operator: "Equal"
      value: "value"
      effect: "NoSchedule"

    affinity:
     nodeAffinity:
       requiredDuringSchedulingIgnoredDuringExecution:
         nodeSelectorTerms:
         - matchExpressions:
           - key: kubernetes.io/e2e-az-name
             operator: In
             values:
             - e2e-az1
```

A compute cluster can be fully occupied running a query, preventing new queries from being
scheduled. To avoid this, you can deploy this chart multiple times in the same cluster, reserving
the resources between the different deployments.
