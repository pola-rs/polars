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
