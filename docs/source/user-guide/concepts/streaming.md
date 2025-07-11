# Streaming

<!-- Not included in the docs “until we have something we are proud of”. https://github.com/pola-rs/polars/pull/19087/files/92bffabe48c6c33a9ec5bc003d8683e59c97158c#r1788988580 -->

One additional benefit of the lazy API is that it allows queries to be executed in a streaming
manner. Instead of processing all the data at once, Polars can execute the query in batches allowing
you to process datasets that do not fit in memory. Besides memory pressure, the streaming engine
also is more performant than Polars' in-memory engine.

To tell Polars we want to execute a query in streaming mode we pass the `engine="streaming"`
argument to `collect`:

{{code_block('user-guide/concepts/streaming','streaming',['collect'])}}

## Inspecting a streaming query

Polars can run many operations in a streaming manner. Some operations are inherently non-streaming,
or are not implemented in a streaming manner (yet). In the latter case, Polars will fall back to the
in-memory engine for those operations. A user doesn't have to know about this, but it can be
interesting for debugging memory or performance issues.

To inspect the physical plan of streaming query, you can plot the physical graph. The legend shows
how memory intensive the operation can be.

```python
--8<-- "python/user-guide/concepts/streaming.py:createplan_query"
```

```python exec="on" session="user-guide/concepts/streaming"
--8<-- "python/user-guide/concepts/streaming.py:createplan"
```
