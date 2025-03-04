# Getting started

Polars Cloud is a managed compute platform for your Polars queries. It allows you to effortlessly
run your local queries in your cloud environment, both in an interactive setting as well as for ETL
or batch jobs. By working in a 'Bring your own Cloud' model the data never leaves your environment.

## Installation

Install the Polars Cloud python library in your environment

```bash
$ pip install polars polars-cloud
```

Create an account and login by running the command below.

```bash
$ pc login
```

## Connect your cloud

Polars Cloud currently exclusively supports AWS as a cloud provider.

Polars Cloud needs permission to manage hardware in your environment. This is done by deploying our
cloudformation template. See our [infrastructure](providers/aws/infra.md) section for more details.

To connect your cloud run:

```bash
$ pc setup workspace -n <YOUR_WORKSPACE_NAME>
```

This redirects you to the browser where you can connect Polars Cloud to your AWS environment.
Alternatively, you can follow the steps in the browser and create the workspace there.

## Run your queries

Now that we are done with the setup, we can start running queries. The general principle here is
writing Polars like you're always used to and calling `.remote()` on your `LazyFrame`. The following
example shows how to create a compute cluster and run a simple Polars query.

{{code_block('polars-cloud/quickstart','general',['ComputeContext','LazyFrameExt'])}}

Let us go through the code line by line. First we need to define the hardware the cluster will run
on. This can be provided in terms of cpu & memory or by specifying the the exact instance type in
AWS.

```python
ctx = pc.ComputeContext(memory=8, cpus=2 , cluster_size=1)
```

Then we write a regular lazy Polars query. In this simple example we compute the maximum of column
`a` over column `b`.

```python
df = pl.LazyFrame({
    "a": [1, 2, 3],
    "b": [4, 4, 5]
})
lf = df.with_columns(
    c = pl.col("a").max().over("b")
)
```

Finally we are going to run our query on the compute cluster. We use `.remote()` to signify that we
want to run the query remotely. This gives back a special version of the `LazyFrame` with extension
methods. Up until this point nothing has executed yet, calling `.write_parquet()` sends the query to
g
