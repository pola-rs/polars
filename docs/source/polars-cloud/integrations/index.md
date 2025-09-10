# Orchestrate your queries

Polars Cloud can be integrated and used in popular third-party orchestration tools. This approach
lets you focus on data logic while Polars Cloud handles query execution, eliminating infrastructure
management.

Consider a typical workflow: generating a daily report by ingesting, transforming, and joining two
datasets. We omit standard Polars code to focus on orchestration-specific steps: defining data flow
and managing [service accounts](/polars-cloud/explain/service-accounts) credentials for Polars Cloud
authentication.

The most commonly used data orchestrators are presented in alphabetical order:

- [Airflow](/polars-cloud/integrations/airflow)
- [Dagster](/polars-cloud/integrations/dagster)
- [Prefect](/polars-cloud/integrations/prefect)

In a similar manner, we suggest a way to orchestrate Polars Cloud queries using AWS-native
infrastructure:

- [AWS lambda](/polars-cloud/integrations/lambda)
