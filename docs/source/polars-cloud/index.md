![Image showing the Polars Cloud logo](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/polars-cloud.svg)

# Introducing Polars Cloud

DataFrame implementations always differed from SQL and databases. SQL could run anywhere from
embedded databases to massive data warehouses. Yet, DataFrame users have been forced to choose
between a solution for local work or solutions geared towards distributed computing, each with their
own APIs and limitations.

Polars is bridging this gap with **Polars Cloud**. Build on top of the popular open source project,
Polars Cloud enables you to write DataFrame code once and run it anywhere. The distributed engine
available with Polars Cloud allows to scale your Polars queries beyond a single machine.

## Key Features of Polars Cloud

- **Unified DataFrame Experience**: Run a Polars query seamlessly on your local machine and at scale
  with our new distributed engine. All from the same API.
- **Serverless Compute**: Effortlessly start compute resources without managing infrastructure with
  options to execute queries on both CPU and GPU (coming soon).
- **Any Environment**: Start a remote query from a notebook on your machine, Airflow DAG, AWS
  Lambda, or your server. Get the flexibility to embed Polars Cloud in any environment.

## Install Polars Cloud

Simply extend the capabilities of Polars with:

```bash
pip install polars polars_cloud
```

## Example query

To run your query in the cloud, simply write Polars queries like you are used to, but call
`LazyFrame.remote()` to indicate that the query should be run remotely.

{{code_block('polars-cloud/index','index',['ComputeContext','LazyFrameRemote'])}}

## Sign up today and start your 30 day trial

Polars Cloud is available to try with a 30 day free trial. You can sign up on
[cloud.pola.rs](https://cloud.pola.rs) to get started.

## Cloud availability

![AWS logo](https://raw.githubusercontent.com/pola-rs/polars-static/refs/heads/master/polars_cloud/aws-logo.svg)

Polars Cloud is available on AWS. Other cloud providers and on-premise solutions are on the roadmap
and will become available in the upcoming months.
