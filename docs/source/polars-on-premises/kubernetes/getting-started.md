First of all, make sure to obtain a license for Polars On-Prem by
[signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). You will receive an access key for
our private Docker registry as well as a JSON-formatted license for running Polars On-Prem.

Polars On-Prem for Kubernetes is distributed through our Helm Chart, which can be found in our
[helm-charts repository](https://github.com/polars-inc/helm-charts/).

## Usage

[Helm](https://helm.sh) must be installed to use the charts. Please refer to Helm
[documentation](https://helm.sh/docs/) to get started. Once Helm is set up properly, add the
repository as follows:

```shell
helm repo add polars-inc https://polars-inc.github.io/helm-charts
helm repo update
```

You can then run `helm search repo polars-inc` to see the available charts.

Create a secret for the received offline license key.

```shell
kubectl create secret generic polars-offline-license --from-file=license.json=license.json
```

For this quickstart, we install a SeaweedFS instance for the output results. The following commands
install the Polars chart and expose the services locally.

```shell
helm upgrade --install polars polars-inc/polars \
    --set license.secretName=polars-offline-license \
    --set license.secretProperty=license.json \
    --set anonymousResults.seaweedfs.enabled=true
kubectl port-forward svc/polars-scheduler 5051:5051
kubectl port-forward svc/polars-observatory 3001:3001
kubectl port-forward svc/polars-seaweedfs 8333:8333
```

You can then run a simple query like so:

```python
import polars as pl
import polars_cloud as pc

ctx = pc.ClusterContext(compute_address="localhost")

result = (
    pl.LazyFrame(
        {
            "a": [1, 2, 3],
            "b": [4, 4, 5],
        }
    )
    .with_columns(
        pl.col("a").max().over("b").alias("c"),
    )
    .remote(ctx)
    .execute()
)

print(result.head)
```

To get your polars cluster ready for production, see the different configuration values. Most
importantly:

- [Anonymous Results](/polars-on-premises/kubernetes/configuration/anonymous-results)
- [Shuffle Data](/polars-on-premises/kubernetes/configuration/shuffle-data)
- [Resource Allocation](/polars-on-premises/kubernetes/configuration/resource-allocation)
