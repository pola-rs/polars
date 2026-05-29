# Azure Kubernetes Service (AKS)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Workload Identity

Through Workload identity, you can securely access private Azure Blob Container data without needing
to manage service account keys or credentials. You could use Microsoft Entra Workload ID for this
purpose.
[See the guide in the official AKS documentation](https://learn.microsoft.com/en-us/azure/aks/workload-identity-deploy-cluster).

```bash
helm upgrade --install polars polars-inc/polars \
  --set scheduler.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
  --set worker.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
# ...
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
# ..
)
```

You may also use Azure Blob Storage as
[an anonymous results location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data)
by configuring the values as such:

```yaml
anonymousResults:
  abs:
    enabled: true
    endpoint: "az://YOUR_BLOB_CONTAINER_NAME/PATH/TO/DATA/"
    options:
    - name: account_name
      value: "YOUR_STORAGE_ACCOUNT_NAME"
```

## Reducing egress costs

!!! warning "Egress charges"

    When Polars workers access Azure Blob Storage (ABS) over the public internet, Azure charges
    data transfer (egress) fees that can grow significantly with large or frequent queries.

Enabling a **Service Endpoint** for `Microsoft.Storage` on your AKS subnet routes all Blob Storage
traffic over the Microsoft backbone network at no additional cost and requires no changes to your
Polars deployment. Once the endpoint is in place and the storage account network rules are updated
to allow access from your subnet, traffic is routed privately and automatically.

Note that the storage account must be in the same region as the AKS cluster (cross-region traffic
incurs charges regardless of the endpoint configuration).

See the related
[guide](https://learn.microsoft.com/en-us/azure/storage/common/storage-network-security) in the
official Azure documentation.
