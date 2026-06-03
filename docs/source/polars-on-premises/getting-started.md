# Getting Started

This quickstart deploys a Polars On-Prem cluster on Kubernetes and runs your first remote query.

**Prerequisites:** a provisioned Kubernetes cluster with `kubectl` and Helm installed.

## Install the client library

```bash
pip install polars polars_cloud
```

### 1. Create an account and workspace

Creating an account and workspace can be done through the
[cloud portal](https://cloud.pola.rs/api/redirects/register) by selecting **Kubernetes** as the
deployment method upon workspace creation. Alternatively use the CLI:

```bash
pc login
pc setup --workspace-type kubernetes
```

### 2. Create a service account

In the workspace settings page, create a service account. Copy the **client ID** and **client
secret**, they are not stored and cannot be retrieved later. Alternatively, you can use the
`polars-cloud` CLI to create a service account.

```bash
polars-cloud service-account create --workspace-name <WORKSPACE NAME> --name ServiceAccount
```

### 3. Deploy with Helm

The example below will deploy a Polars cluster with 2 worker nodes each with 4Gi of memory and a
temporary storage bucket. Replace Workspace ID and Service Account credentials with your own.

!!! info "Workspace ID"

    The Workspace ID can be found in the workspace settings page or with `pc workspace list`.

```bash
helm repo add polars-inc https://polars-inc.github.io/helm-charts && helm repo update
```

```bash
helm upgrade --install polars polars-inc/polars \
  --set clusterId="My First Cluster" \
  --set license.onPrem.enabled=true \
  --set license.onPrem.workspaceId=<WORKSPACE ID> \
  --set license.onPrem.clientId=<SERVICE ACCOUNT ID> \
  --set license.onPrem.clientSecret=<SERVICE ACCOUNT SECRET> \
  --set scheduler.deployment.runtimeContainer.resources.requests.memory=1Gi \
  --set worker.deployment.replicaCount=2 \
  --set worker.deployment.runtimeContainer.resources.requests.memory=4Gi \
  --set worker.deployment.runtimeContainer.resources.limits.memory=4Gi \
  --set anonymousResults.temporaryStorage.enabled=true
```

!!! warning "Not for production use"

    The cluster configuration defined above is for a quickstart only and should not be used in a
    production environment! See the [Kubernetes deployment guide](./kubernetes/index.md)
    for more details.

Verify all pods are running before continuing:

```bash
kubectl get pods
```

```
NAME                                         READY   STATUS    RESTARTS   AGE
polars-scheduler-xxxxxxxxx-xxxxx             1/1     Running   0          1m
polars-worker-xxxxxxxxx-xxxxx                1/1     Running   0          1m
polars-worker-xxxxxxxxx-xxxxx                1/1     Running   0          1m
polars-temporary-storage-xxxxxxxxx-xxxxx     1/1     Running   0          1m
```

## 4. Run your first query

Port-forward the required services, each in a separate terminal:

```bash
kubectl port-forward svc/polars-scheduler 5051:5051
```

```bash
kubectl port-forward svc/polars-observatory 3001:3001
```

```bash
kubectl port-forward svc/polars-temporary-storage 8333:8333
```

Then submit a query from a script or notebook cell:

```python
import polars as pl
import polars_cloud as pc

ctx = pc.ClusterContext(compute_address="localhost")
result = (
    pl.LazyFrame()
    .with_columns(a=pl.arange(0, 100000000).sum())
    .remote(ctx)
    .execute()
)
print(result.head())
```

Your cluster is ready. For a full walkthrough of deployed resources, networking, and production
configuration, see the [Kubernetes deployment guide](./kubernetes/index.md).
