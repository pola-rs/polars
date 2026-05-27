# Azure Kubernetes Service (AKS)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Workload Identity

Through Workload identity, you can securely access private Azure Blob Container data without needing to manage service account keys or credentials. You could use Microsoft Entra Workload ID for this purpose. [See the guide in the official AKS documentation](https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster).

```bash
helm upgrade --install polars polars-inc/polars \
  --set scheduler.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
  --set worker.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
# ...
```

Assuming you have a Blob Container already set up (see quick-start [here](https://learn.microsoft.com/en-us/azure/storage/blobs/storage-quickstart-blobs-portal#create-a-container)), you can then scan or sink directly from the bucket.

```python
path = f"az://YOUR_BLOB_CONTAINER_NAME/PATH/TO/DATA/"
storage_options = {
    "account_name": "YOUR_STORAGE_ACCOUNT_NAME",
}
q = (
    pl.scan_parquet(path, storage_options=storage_options)
# ..
)
```

You may also use Azure Blob Storage as [an anonymous results location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data) b configuring the values as such:

```yaml
anonymousResults:
  abs:
    enabled: true
    endpoint: "az://YOUR_BLOB_CONTAINER_NAME/PATH/TO/DATA/"
    options:
    - name: account_name
      value: "YOUR_STORAGE_ACCOUNT_NAME"
```
