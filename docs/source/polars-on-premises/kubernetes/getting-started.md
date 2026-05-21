Polars On-Prem for Kubernetes is distributed through our Helm Chart, which can be found in our
[helm-charts repository](https://github.com/polars-inc/helm-charts/).

## Usage

[Helm](https://helm.sh) must be installed to use the charts. Please refer to Helm
[documentation](https://helm.sh/docs/) to get started. Once Helm is set up properly, add the
repository as follows:

```shell
helm repo add polars-inc https://polars-inc.github.io/helm-charts
```

You can then run `helm search repo polars-inc` to see the available charts.

Further explanation on the different configuration can be found in the
[chart `README.md`](https://github.com/polars-inc/helm-charts/blob/main/charts/polars/README.md).
