# Debugging with pipes

Suppose that you write a long chain of transformations:

{{code_block('user-guide/misc/debugging_with_pipes','pipeline1',[])}}

```python exec="on" session="user-guide/misc/debugging_with_pipes"
--8<-- "python/user-guide/misc/debugging_with_pipes.py:setup"
```

```python exec="on" session="user-guide/misc/debugging_with_pipes"
--8<-- "python/user-guide/misc/debugging_with_pipes.py:pipeline1"
```

... and in the middle of the chain something breaks.

How do you insert `print` and `assert` statements into the middle of the chain?

Consider writing your own helper functions and saving them
(as you might need them multiple times in the future). For example:

{{code_block('user-guide/misc/debugging_with_pipes','assert_schema',[])}}

```python exec="on" session="user-guide/misc/debugging_with_pipes"
--8<-- "python/user-guide/misc/debugging_with_pipes.py:assert_schema"
```

{{code_block('user-guide/misc/debugging_with_pipes','print_expr',[])}}

```python exec="on" session="user-guide/misc/debugging_with_pipes"
--8<-- "python/user-guide/misc/debugging_with_pipes.py:print_expr"
```

Now you can insert a couple of lines here:

{{code_block('user-guide/misc/debugging_with_pipes','pipeline2',[])}}

```python exec="on" result="text" session="user-guide/misc/debugging_with_pipes"
--8<-- "python/user-guide/misc/debugging_with_pipes.py:pipeline2"
```

When your debugging session is over, you can remove those lines.
