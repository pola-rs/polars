## Dealing with multiple files.

Polars can deal with multiple files differently depending on your needs and memory strain.

Let's create some files to give us some context:

{{code_block('user-guide/io/multiple','create',['write_csv'])}}

## Reading into a single `DataFrame`

To read multiple files into a single `DataFrame`, we can use globbing patterns:

{{code_block('user-guide/io/multiple','read',['read_csv'])}}

```python exec="on" result="text" session="user-guide/io/multiple"
--8<-- "python/user-guide/io/multiple.py:create"
--8<-- "python/user-guide/io/multiple.py:read"
```

To see how this works we can take a look at the query plan. Below we see that all files are read separately and
concatenated into a single `DataFrame`. Polars will try to parallelize the reading.

{{code_block('user-guide/io/multiple','graph',['show_graph'])}}

```python exec="on" session="user-guide/io/multiple"
--8<-- "python/user-guide/io/multiple.py:creategraph"
```

## Reading and processing in parallel

If your files don't have to be in a single table you can also build a query plan for each file and execute them in parallel
on the Polars thread pool.

All query plan execution is embarrassingly parallel and doesn't require any communication.

{{code_block('user-guide/io/multiple','glob',['scan_csv'])}}

```python exec="on" result="text" session="user-guide/io/multiple"
--8<-- "python/user-guide/io/multiple.py:glob"
```
