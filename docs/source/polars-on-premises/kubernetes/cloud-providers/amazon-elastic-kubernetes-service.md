# Amazon Elastic Kubernetes Service (EKS)

!!! info "Initial configuration"

    This page expects that you've already set up a Polars cluster once through the Polars Cloud onboarding or the [getting started guide](../index.md).

## Data access using Pod Identity

Through Pod Identity, you can securely access private S3 buckets without needing to manage service
account keys or credentials.
[See the guide in the official EKS documentation](https://docs.aws.amazon.com/eks/latest/userguide/service-accounts.html).

```bash
helm upgrade --install polars polars-inc/polars \
  --set scheduler.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
  --set worker.serviceAccount.name=<YOUR_SERVICE_ACCOUNT_NAME> \
# ...
```

Assuming you have an S3 bucket already set up (see quick-start
[here](https://docs.aws.amazon.com/AmazonS3/latest/userguide/create-bucket-overview.html)), you can
then scan or sink directly from the bucket.

```python
path = f"s3://YOUR_S3_BUCKET_NAME/PATH/TO/DATA/"

q = (
    pl.scan_parquet(path)
# ...
)
```

You may also use S3 as
[an anonymous results location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#anonymous-results-data)
by configuring the values as such:

```yaml
anonymousResults:
  s3:
    enabled: true
    endpoint: "s3://YOUR_S3_BUCKET_NAME/PATH/TO/DATA/"
```

To use S3 as
[a shuffle location](https://github.com/polars-inc/helm-charts/tree/main/charts/polars#shuffle-data),
configure the values as such:

```yaml
shuffleData:
  s3:
    enabled: true
    endpoint: "s3://YOUR_S3_BUCKET_NAME/PATH/TO/DATA/"
```

## Reducing NAT costs for private nodes

In clusters where nodes have no public IP addresses, S3 traffic is typically routed through a NAT
Gateway, billed according to the volume of data processed. Creating a **VPC Gateway Endpoint for
S3** bypasses the NAT Gateway entirely: S3 traffic is then routed directly over the AWS private
network at no additional cost, with no changes required to your Polars deployment.

Note that the S3 bucket must be in the same region as the EKS cluster (cross-region traffic incurs
charges regardless of the endpoint configuration).

See the related [guide](https://docs.aws.amazon.com/vpc/latest/privatelink/vpc-endpoints-s3.html) in
the official AWS documentation.
