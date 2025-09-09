# AWS Lambda

In this small set of paragraphs we draft a solution to orchestrate Polars cloud queries through AWS-native infrastructure, that is, AWS EventBridge and AWS Lambda.
The former to trigger the latter, which in turn submits a query to Polars Cloud.

!!! tip
    Submitting a query does not require for the process submitting it to remain alive if the Polars Cloud [compute context](/polars-cloud/context/compute-context) is **\*not\*** built as a regular `Python` context manager.

## Lambda function

The first hurdle is providing an environment including `Polars` dependencies to the lambda function; this can be done in various ways, all documented by AWS [here](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html).
The most commonly used approach is via [creating a `zip` package including dependencies](https://docs.aws.amazon.com/lambda/latest/dg/python-package.html#python-package-create-dependencies).
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

Once the query is submitted the lambda will gracefully exit, leaving the rest of the handling to Polars Cloud.

## Triggering rule

Since we are here not using any dedicated orchestrator infrastructure (like [Airflow](/polars-cloud/integrations/airflow) for instance) we can instead generate triggering rules in AWS EventBridge.
Rules can be defined via the AWS Console (point-and-click) or via the AWS CLI, as documented [here](https://docs.aws.amazon.com/eventbridge/latest/userguide/eb-run-lambda-schedule.html).
A simple CRON rule should be enough to trigger your query to run at given interval.
