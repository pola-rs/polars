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
