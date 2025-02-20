# Interactive and batch mode

In Polars Cloud a user can define two types of compute modes: batch & interactive. Batch mode is designed for job style queries. These kind of queries are typically scheduled and run once in a certain period. Interactive mode allows for exploratory workflows where a user interacts with the dataset and requires more compute resources than locally available.

The rest of this page will give examples on how to set up one or the other. More information on the architectural differences and implications can be found on [the infrastructure page](../../providers/aws/infra).

## Example

{{code_block('polars-cloud/interactive-batch','example',[])}}

## Batch

Batch workflows are systematic data processing pipelines, written for scheduled execution. They typically process large volumes of dat in scheduled intervals (e.g. hourly, daily, etc.). Key characteristic is that the executed job has a defined lifetime. A predefined compute instance should spin up at a certain time and shutdown when the job is executed.

Polars Cloud makes it easy to run your query at scale whenever your use case requires it. You can develop your query on your local machine and define a compute context and destination to execute it in your cloud environment.

{{code_block('polars-cloud/interactive-batch','batch',['ComputeContext'])}}

## Interactive

Interactive data workflows are more iterative and discovery focused workflows. These workflows have a more ad-hoc nature that evolves as insights are discovered. It is often used by data scientists to explore new features for their models or by data analysts to uncover patterns.

The difference with batch data workflows is that a shorter feedback cycle is expected from the user, as they want to inspect the result and continu their exploration. Polars Cloud support this interactive workflow by supporting an interactive mode. In this mode, users can start one or more compute machine(s) to explore larger datasets that do not fit on their local development machine.

As this mode will be used for exploratory use cases and short feedback cycles the queries are not logged to Polars Cloud and will not be availble for later inspection.

{{code_block('polars-cloud/interactive-batch','interactive',[])}}

The initial query remaims the same. In the compute context the parameter `interactive` should be set to `True`. This will ensure that the machine remains available until the user stops it.

```python
print(res1)

```

```text
total_stages: 1
finished_stages: 1
total_rows: 4
location: ['s3://bucket/output.parquet']
head:
┌────────────────┬────────────┬───────────┐
│ name           ┆ birth_year ┆ bmi       │
│ ---            ┆ ---        ┆ ---       │
│ str            ┆ i32        ┆ f64       │
╞════════════════╪════════════╪═══════════╡
│ Chloe Cooper   ┆ 1983       ┆ 19.687787 │
│ Ben Brown      ┆ 1985       ┆ 23.141498 │
│ Alice Archer   ┆ 1997       ┆ 23.791913 │
│ Daniel Donovan ┆ 1981       ┆ 27.134694 │
└────────────────┴────────────┴───────────┘
```

To continue your query we can read the result to a new LazyFrame and continue your exploration.

{{code_block('polars-cloud/interactive-batch','interactive-next',[])}}

```python
print(res2)
```

```text
total_stages: 1
finished_stages: 1
total_rows: 2
location: ['s3://bucket/output2.parquet']
head:
┌──────────────┬────────────┬───────────┐
│ name         ┆ birth_year ┆ bmi       │
│ ---          ┆ ---        ┆ ---       │
│ str          ┆ i32        ┆ f64       │
╞══════════════╪════════════╪═══════════╡
│ Chloe Cooper ┆ 1983       ┆ 19.687787 │
│ Ben Brown    ┆ 1985       ┆ 23.141498 │
└──────────────┴────────────┴───────────┘
```
