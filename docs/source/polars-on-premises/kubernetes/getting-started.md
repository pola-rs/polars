First of all, make sure to obtain a license for Polars on-premises by
[signing up here](https://w0lzyfh2w8o.typeform.com/to/zuoDgoMv). You will receive an access key for
our private Docker registry as well as a license for running Polars Distributed on-premise.

Polars-on-premises for Kubernetes is distributed through our Helm Chart, which can be found in our
[helm-charts repository](https://github.com/polars-inc/helm-charts/).

## Usage

[Helm](https://helm.sh) must be installed to use the charts. Please refer to Helm's
[documentation](https://helm.sh/docs/) to get started.

Once Helm is set up properly, add the repository as follows:

```console
helm repo add polars-inc https://polars-inc.github.io/helm-charts
```

You can then run `helm search repo polars-inc` to see the charts.

Further explanation on the different configuration can be found in the
[Helm Chart's readme](https://github.com/polars-inc/helm-charts/blob/main/charts/polars/README.md).
