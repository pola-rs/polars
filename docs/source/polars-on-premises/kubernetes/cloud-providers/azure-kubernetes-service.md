# Azure Kubernetes Service (AKS)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Workload Identity

Through Workload identity, you can securely access private Azure Blob Container data without needing
to manage service account keys or credentials. You could use Microsoft Entra Workload ID for this
purpose.
[See the guide in the official AKS documentation](https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster).

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

Assuming you have a Blob Container already set up (see quick-start
[here](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal#create-a-container)),
you can then scan or sink directly from the bucket.

```python
path = f"az://YOUR_BLOB_CONTAINER_NAME/PATH/TO/DATA/"
storage_options = {
    "account_name": "YOUR_STORAGE_ACCOUNT_NAME",
}
q = (
    pl.scan_parquet(path, storage_options=storage_options)
# ...
)
```

You may also use Azure Blob Storage as an anonymous results location by configuring the values as
such:

=== "Helm Chart"

    ```yaml
    anonymousResults:
      abs:
        enabled: true
        endpoint: "az://YOUR_BLOB_CONTAINER_NAME/PATH/TO/DATA/"
        options:
        - name: account_name
          value: "YOUR_STORAGE_ACCOUNT_NAME"
    ```

    See the [`anonymousResults` section](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data)
    of the Helm chart values.

=== "Kubernetes Operator"

    ```yaml
    anonymousResults:
      abs:
        endpoint: "az://YOUR_BLOB_CONTAINER_NAME/PATH/TO/DATA/"
        options:
        - name: account_name
          value: "YOUR_STORAGE_ACCOUNT_NAME"
    ```

    See [`AnonymousResultsABSSpec`](https://github.com/polars-inc/polars-k8s-operator/blob/main/docs/api.md#anonymousresultsabsspec)
    in the operator's CRD reference.
