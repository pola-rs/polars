# Google BigQuery

To read or write from GBQ, additional dependencies are needed:

=== ":fontawesome-brands-python: Python"

```shell
$ pip install google-cloud-bigquery
```

## Read

We can load a query into a `DataFrame` like this:

{{code_block('user-guide/io/bigquery','read',['from_arrow'])}}

## Write

{{code_block('user-guide/io/bigquery','write',[])}}

## Scan

Polars allows you to _scan_ a BigQuery table. Scanning delays the actual
reading of the table and instead returns a lazy computation holder called a
`LazyFrame`. Many filters can be pushed down to the BigQuery Storage Read API
to limit the amount of data downloaded.


{{code_block('user-guide/io/bigquery','scan',['scan_bigquery'])}}

If you want to know why this is desirable, you can read more about these Polars optimizations
[here](../concepts/lazy-api.md).
