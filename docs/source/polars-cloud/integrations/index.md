# Orchestrate your queries

In these sections we imagine a simple scenario: two datasets that need to be ingested (with
cleanup/transformations) and joined together as part of a report. Most of the `Polars` code will be
omitted for clarity, and to focus on what is relevant here: the overall data flow including Polars
Cloud, and secret management; the only secret being the
[service account](/polars-cloud/explain/service-accounts) used to authenticate to Polars Cloud.

The most commonly used data orchestrators are presented in alphabetical order:

- [Airflow](/polars-cloud/integrations/airflow)
- [Dagster](/polars-cloud/integrations/dagster)
- [Prefect](/polars-cloud/integrations/prefect)

In a similar manner, we suggest a way to orchestrate Polars Cloud queries using AWS-native
infrastructure:

- [AWS lambda](/polars-cloud/integrations/lambda)
