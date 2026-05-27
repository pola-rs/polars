# Azure Kubernetes Service (EKS)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Pod Identity

Through Pod Identity, you can securely access private S3 buckets without needing to manage service account keys or credentials. [See the guide in the official EKS documentation](https://docs.aws.amazon.com/eks/latest/userguide/service-accounts.html).

```bash
helm upgrade --install polars polars-inc/polars \
  --set scheduler.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
  --set worker.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
# ...
```

Assuming you have an S3 bucket already set up (see quick-start [here](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)), you can then scan or sink directly from the bucket.

```python
path = f"s3://YOUR_S3_BUCKET_NAME/PATH/TO/DATA/"

q = (
    pl.scan_parquet(path)
# ..
)
```

You may also use S3 as [an anonymous results location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data) by configuring the values as such:

```yaml
anonymousResults:
  s3:
    enabled: true
    endpoint: "s3://YOUR_S3_BUCKET_NAME/PATH/TO/DATA/"
```

To use S3 as [a shuffle location]([an anonymous results location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#shuffle-data), configure the values as such:

```yaml
shuffleData:
  s3:
    enabled: true
    endpoint: "s3://YOUR_S3_BUCKET_NAME/PATH/TO/DATA/"
```
