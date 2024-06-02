# Building abstractions with pipes

All programming languages (e.g. Rust or Python) provide some _primitive operations_ (e.g. `+` or `sqrt`), some means of _combining_ them into complex pipelines, and some means of _hiding complexity behind abstractions_. An abstraction  (e.g. a named function) is a simple name for a piece of complex code.

The API of Polars is a small domain-specific language. This language cannot (and should not) accomodate all needs with an ever-growing vocabulary of primitive operations. Instead it gives you the tools to build your own abstractions.

```python exec="on" session="user-guide/pipes/building_abstractions_with_pipes"
--8<-- "python/user-guide/pipes/building_abstractions_with_pipes.py:setup"
```

Suppose, for example, that you frequently have to apply the Pythagorean theorem to your data. Create a function for that:

{{code_block('user-guide/pipes/building_abstractions_with_pipes','hypothenuse',[])}}

```python exec="on" session="user-guide/pipes/building_abstractions_with_pipes"
--8<-- "python/user-guide/pipes/building_abstractions_with_pipes.py:hypothenuse"
```

... and apply it with `pipe`:

{{code_block('user-guide/pipes/building_abstractions_with_pipes','pipe',['pipe'])}}

```python exec="on" result="text" session="user-guide/pipes/building_abstractions_with_pipes"
--8<-- "python/user-guide/pipes/building_abstractions_with_pipes.py:pipe"
```

