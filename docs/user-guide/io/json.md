# JSON files

Polars can read and write both standard JSON and newline-delimited JSON (NDJSON).

## Read

### JSON

Reading a JSON file should look familiar:

{{code_block('user-guide/io/json','read',['read_json'])}}

### Newline Delimited JSON

JSON objects that are delimited by newlines can be read into polars in a much more performant way than standard json.

Polars can read an NDJSON file into a `DataFrame` using the `read_ndjson` function:

{{code_block('user-guide/io/json','readnd',['read_ndjson'])}}

## Write

{{code_block('user-guide/io/json','write',['write_json','write_ndjson'])}}

## Scan

Polars allows you to _scan_ a JSON input **only for newline delimited json**. Scanning delays the actual parsing of the
file and instead returns a lazy computation holder called a `LazyFrame`.

{{code_block('user-guide/io/json','scan',['scan_ndjson'])}}
