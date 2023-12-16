# CSV

## Read & write

Reading a CSV file should look familiar:

{{code_block('user-guide/io/csv','read',['read_csv'])}}

Writing a CSV file is similar with the `write_csv` function:

{{code_block('user-guide/io/csv','write',['write_csv'])}}

## Scan

Polars allows you to _scan_ a CSV input. Scanning delays the actual parsing of the
file and instead returns a lazy computation holder called a `LazyFrame`.

{{code_block('user-guide/io/csv','scan',['scan_csv'])}}

If you want to know why this is desirable, you can read more about these Polars
optimizations [here](../concepts/lazy-vs-eager.md).
