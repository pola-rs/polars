# Styling

Data in a Polars `DataFrame` can be styled for presentation use the `DataFrame.style` property. This returns a `GT` object from [Great Tables](https://posit-dev.github.io/great-tables/articles/intro.html), which enables structuring, formatting, and styling for table display.

{{code_block('user-guide/misc/styling','dataframe',[])}}

```python exec="on" result="text" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:dataframe"
```

## Structure: add header title

{{code_block('user-guide/misc/styling','structure-header',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:structure-header-out"
```

## Structure: add row stub

{{code_block('user-guide/misc/styling','structure-stub',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:structure-stub-out"
```

## Structure: add column spanner

{{code_block('user-guide/misc/styling','structure-spanner',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:structure-spanner-out"
```

## Format: limit decimal places

{{code_block('user-guide/misc/styling','format-number',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:format-number-out"
```

## Style: highlight max row

{{code_block('user-guide/misc/styling','style-simple',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:style-simple-out"
```

## Style: bold species column

{{code_block('user-guide/misc/styling','style-bold-column',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:style-bold-column-out"
```

## Full example

{{code_block('user-guide/misc/styling','full-example',[])}}

```python exec="on" session="user-guide/misc/styling"
--8<-- "python/user-guide/misc/styling.py:full-example-out"
```
