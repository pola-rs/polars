# FAQ

On this page you can find answers to some frequently asked questions around Polars Cloud.

## Who is behind Polars Cloud?

Polars Cloud is built by the organization behind the open source Polars project. We are committed to
improve Polars open source for all single machine workloads. Polars Cloud will extend Polars
functionalities for remote and distributed compute.

## Where does the compute run?

All compute runs in your own cloud environment. The main reason is that this ensures that your data
never leaves your environment and that the compute is always close to your data.

You can learn more about how this setup in
[the infrastructure section of the documentation](providers/aws/infra.md).

## Can you run Polars Cloud on-premise?

Currently, Polars Cloud is only available to organizations that are on AWS. Support for on-premise
infrastructure is on our roadmap and will become available soon.

## What does Polars Cloud offer me beyond Polars?

Polars Cloud offers a managed service that enables scalable data processing with the flexibility and
expressiveness of the Polars API. It extends the open source Polars project with the following
capabilities:

- Distributed engine to scale workloads horizontally.
- Cost-optimized serverless architecture that automatically scales compute resources
- Built-in fault tolerance mechanisms ensuring query completion even during hardware failures or
  system interruptions
- Comprehensive monitoring and analytics tools providing detailed insights into query performance
  and resource utilization.

## What are the main use cases for Polars Cloud?

Polars Cloud offers both a batch as an interactive mode to users. Batch mode can be used for ETL
workloads or one-off large scale analytic jobs. Interactive mode is for users that are looking to do
data exploration on a larger scale data processing that requires more compute than their own machine
can offer.

## How can Polars Cloud integrate with my workflow?

One of our key priorities is ensuring that running remote queries feels as native and seamless as
running them locally. Every user should be able to scale their queries effortlessly.

Polars Cloud is completely environment agnostic. This allows you to run your queries from anywhere
such as your own machine, Jupyter/Marimo notebooks, Airflow DAGs, AWS Lambda functions, or your
servers. By not tying you to a specific platform, Polars Cloud gives you the flexibility to execute
your queries wherever it best fits your workflow.

## What is the pricing model of Polars Cloud?

Polars Cloud is available at no additional cost in this early stage. You only pay for the resources
you use in your own cloud environment. We are exploring different usage based pricing models that
are geared towards running queries as fast and efficient as possible.

## Will the distributed engine be available in open source?

The distributed engine is only available in Polars Cloud. There are no plans to make it available in
the open source project. Polars is focused on single node compute, as it makes efficient use of the
available resources. Users already report utilizing Polars to process hundreds of gigabytes of data
on single (large) compute instance. The distributed engine is geared towards teams and organizations
that are I/O bound or want to scale their Polars queries beyond single machines.
