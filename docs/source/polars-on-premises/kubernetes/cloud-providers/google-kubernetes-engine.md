# Google Kubernetes Engine (GKE)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Workload Identity

Through GKE Workload Identity, you can securely access private Google Cloud Storage (GCS) buckets
without needing to manage service account keys or credentials. In most scenarios, it comes down to
enabling Workload Identity Federation, creating a Kubernetes service account, and creating an IAM
policy binding.
[See the guide in the official GKE documentation](https://docs.cloud.google.com/kubernetes-engine/docs/how-to/workload-identity).

=== "Helm Chart"

    ```yaml
    scheduler:
      serviceAccount:
        name: <YOUR_SERVICE_ACCOUNT_NAME>
    worker:
      serviceAccount:
        name: <YOUR_SERVICE_ACCOUNT_NAME>
    # ...
    ```

=== "Kubernetes Operator"

    ```yaml
    scheduler:
      serviceAccount: {}
    workerPool:
      serviceAccount: {}
    ```

Assuming you have a bucket already set up (see quick-start
[here](https://docs.cloud.google.com/storage/docs/creating-buckets)), you can then scan or sink
directly from the bucket.

```python
path = f"gs://YOUR_BUCKET_NAME/PATH/TO/DATA/"
storage_options = {
    "project": "YOUR_PROJECT_NAME",
}
q = (
    pl.scan_parquet(path, storage_options=storage_options)
# ...
)
```

You may also use Google Cloud Storage as an anonymous results location by configuring the values as
such:

=== "Helm Chart"

    ```yaml
    anonymousResults:
      gcs:
        enabled: true
        endpoint: "gs://YOUR_BUCKET_NAME/PATH/TO/DATA/"
        options:
        - name: project
          value: "YOUR_PROJECT_NAME"
    ```

    See the [`anonymousResults` section](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data)
    of the Helm chart values.

=== "Kubernetes Operator"

    ```yaml
    anonymousResults:
      gcs:
        endpoint: "gs://YOUR_BUCKET_NAME/PATH/TO/DATA/"
        options:
        - name: project
          value: "YOUR_PROJECT_NAME"
    ```

    See [`AnonymousResultsGCSSpec`](https://github.com/polars-inc/polars-k8s-operator/blob/main/docs/api.md#anonymousresultsgcsspec)
    in the operator's CRD reference.

## Accessing GCS from private nodes

In clusters where nodes have no external IP addresses, GCS is unreachable without Cloud NAT; if the
latter is deployed, GCS traffic is billed according to the volume of data processed. Enabling
**Private Google Access** (PGA) on your GKE node subnet gives internally-addressed nodes a direct
path to Google APIs (including GCS) and requires no changes to your Polars deployment. In case both
Cloud NAT and PGA are deployed the latter takes precedence for Google APIs and no costs are
incurred.

Note that the GCS bucket must be in the same region as the GKE cluster (cross-region traffic incurs
charges regardless of whether Private Google Access is enabled).

See the related [guide](https://docs.cloud.google.com/vpc/docs/configure-private-google-access) in
the official GCP documentation.
