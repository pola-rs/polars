# Getting started

Polars Cloud is a managed compute platform for your Polars queries. It allows you to effortlessly run your local queries in your cloud environment, both in an interactive setting as well as for ETL or batch jobs. By working in a 'Bring your own Cloud' model the data never leaves your environment.

## Installation

Install the Polars Cloud python library in your environment

``` bash
pip install polars polars-cloud
```


Create an account and login by running the command below. 

``` bash
plc login
```

## Connect your cloud

Polars Cloud currently supports AWS as a cloud provider. If you are on Azure, GCP or prefer an on-premise solution please contact us directly. 

Polars Cloud needs permission to spin up & down hardware in your environment. This is done by deploying our cloudformation template. See our [infrastructure](providers/aws.md) section for more details.

To connect your cloud run:

``` bash
plc setup workspace -n <YOUR_WORKSPACE_NAME>
```

This redirects you to the browser where you can connect Polars to your AWS environment.

## Run your queries

Now that we are done with the setup, we can start running queries. The following example shows how to create a compute cluster and run a simple Polars query.

=== ":fontawesome-brands-python: Python"

    ``` python
    import polars_cloud as plc
    import polars as pl

    ctx = plc.ComputeContext(memory = 8, cpus = 2 , cluster_size = 1)
    df = pl.DataFrame({"a": [1, 2, 3], "b": [4, 4, 5]})
    lf = df.lazy().with_columns(pl.col("a").max().over("b").alias("c"))
    lf.remote(context = ctx).write_parquet(uri="s3://my-bucket/result.parquet")
    ```


