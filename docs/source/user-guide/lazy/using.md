# Usage

With the lazy API, Polars doesn't run each query line-by-line but instead processes the full query end-to-end. To get the most out of Polars it is important that you use the lazy API because:

- the lazy API allows Polars to apply automatic query optimization with the query optimizer
- the lazy API allows you to work with larger than memory datasets using streaming
- the lazy API can catch schema errors before processing the data

Here we see how to use the lazy API starting from either a file or an existing `DataFrame`.

## Using the lazy API from a file

In the ideal case we would use the lazy API right from a file as the query optimizer may help us to reduce the amount of data we read from the file.

We create a lazy query from the Reddit CSV data and apply some transformations.

By starting the query with `pl.scan_csv` we are using the lazy API.

{{code_block('user-guide/lazy/using','dataframe',['scan_csv','with_columns','filter','col'])}}

A `pl.scan_` function is available for a number of file types including CSV, IPC, Parquet and JSON.

In this query we tell Polars that we want to:

- load data from the Reddit CSV file
- convert the `name` column to uppercase
- apply a filter to the `comment_karma` column

The lazy query will not be executed at this point. See this page on [executing lazy queries](execution.md) for more on running lazy queries.

## Using the lazy API from a `DataFrame`

An alternative way to access the lazy API is to call `.lazy` on a `DataFrame` that has already been created in memory.

{{code_block('user-guide/lazy/using','fromdf',['lazy'])}}

By calling `.lazy` we convert the `DataFrame` to a `LazyFrame`.
