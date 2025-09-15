# AWS Lambda

Orchestrate Polars Cloud queries using AWS-native serverless infrastructure through EventBridge and
Lambda. This section details how to implement scheduled query execution without infrastructure
management by submitting workloads to Polars Cloud via Lambda functions while leveraging AWS Secrets
Manager for secure credential handling.

!!! tip

    Submitting a query does not require for the process submitting it to remain alive if the Polars Cloud [compute context](/polars-cloud/context/compute-context) is **\*not\*** built as a regular `Python` context manager.

## Lambda function

The first hurdle is providing an environment including `Polars` dependencies to the Lambda function;
this can be done in various ways, all documented by AWS
[here](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html). The most commonly used
approach is via
[creating a `zip` package including dependencies](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html#python-package-create-dependencies).
The code for the lambda function can be boiled down to the following (pseudo-code):

```python
import boto3
import polars as pl
import polars_cloud as pc

client = boto3.client("secretsmanager")

# authenticate to polars cloud with the secrets created above
pc.authenticate(
    client_id=client.get_secret_value(SecretId="<SECRET>").get("SecretString"),
    client_secret=client.get_secret_value(SecretId="<SECRET>").get("SecretString"),
)

# define the compute context
cc = pc.ComputeContext(cpus=2, memory=4)

# submit the query
pl.scan_csv(...).remote(cc).sink_parquet(...)
```

Once the query is submitted the Lambda will gracefully exit, leaving the rest of the handling to
Polars Cloud.

## Triggering rule

Since we are here not using any dedicated orchestrator infrastructure (like [Airflow](airflow.md)
for instance) we can instead generate triggering rules in AWS EventBridge. Rules can be defined via
the AWS Console (point-and-click) or via the AWS CLI, as documented
[here](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-run-lambda-schedule.html). A
simple CRON rule should be enough to trigger your query to run at given interval.
