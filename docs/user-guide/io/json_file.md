# JSON files

## Read & write

### JSON

Reading a JSON file should look familiar:

{{code_block('user-guide/io/json-file','read',['read_json'])}}

### Newline Delimited JSON

JSON objects that are delimited by newlines can be read into polars in a much more performant way than standard json.

{{code_block('user-guide/io/json-file','readnd',['read_ndjson'])}}

## Write

{{code_block('user-guide/io/json-file','write',['write_json','write_ndjson'])}}

## Scan

`Polars` allows you to _scan_ a JSON input **only for newline delimited json**. Scanning delays the actual parsing of the
file and instead returns a lazy computation holder called a `LazyFrame`.

{{code_block('user-guide/io/json-file','scan',['scan_ndjson'])}}
