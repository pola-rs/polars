# Permissions

The workspace is an isolation for all resources living within your cloud environment. Every
workspace has a single instance profile which defines the permissions for the compute. This profile
is attached to the compute within your environment. By default, the profile can read and write from
S3, but you can easily adjust depending on your own infrastructure stack.

## Adding or removing permissions

If you want Polars Cloud to be able to read from other data sources than `S3` within your cloud
environment you must provide the access control from directly within AWS. To do this go to `IAM`
within the aws console and locate the role called `polars-<WORKSPACE_NAME>-IAMWorkerRole-<slug>`.
Here you can adjust the permissions of the workspace for instance:

- [Narrow down the S3 access to certain buckets](https://docs.aws.amazon.com/IAM/latest/UserGuide/reference_policies_examples_s3_deny-except-bucket.html)
- [Provide IAM access to rds database](https://docs.aws.amazon.com/AmazonRDS/latest/UserGuide/UsingWithRDS.IAMDBAuth.IAMPolicy.html)

## Assuming a different role

To use a different IAM role with Polars Cloud beyond the default
`polars-<WORKSPACE_NAME>-IAMWorkerRole-<slug>`, you need to configure cross-account role assumption.
Set up your target role's trust policy to allow the Polars Cloud role to assume it by following the
[AWS cross-account role documentation](https://docs.aws.amazon.com/IAM/latest/UserGuide/access_policies-cross-account-resource-access.html#access_policies-cross-account-using-roles).
After configuring the trust relationship, specify the role ARN in the storage_options parameter when
reading or writing data to have Polars assume that role for the operation.

```python
import polars_cloud as pc
import polars as pl

ctx = pc.ComputeContext(cpus=1, memory=1)
lf = pl.scan_parquet(
    "s3://your-bucket/foo.parquet",
    credential_provider=pl.CredentialProviderAWS(
        assume_role={
            "RoleArn": "<INSERT-DESIRED-ARN-HERE>"
            "RoleSessionName": "AssumedRole"
        },
    ),
)
```
