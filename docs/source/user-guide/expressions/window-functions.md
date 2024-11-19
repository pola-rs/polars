# Window functions

Window functions are expressions with superpowers. They allow you to perform aggregations on groups
within the context `select`. Let's get a feel for what that means.

First, we load a Pokémon dataset:

{{code_block('user-guide/expressions/window','pokemon',['read_csv'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:pokemon"
```

## Operations per group

Window functions are ideal when we want to perform an operation within a group. For instance,
suppose we want to rank our Pokémon by the column “Speed”. However, instead of a global ranking, we
want to rank the speed within each group defined by the column “Type 1”. We write the expression to
rank the data by the column “Speed” and then we add the function `over` to specify that this should
happen over the unique values of the column “Type 1”:

{{code_block('user-guide/expressions/window','rank',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:rank"
```

To help visualise this operation, you may imagine that Polars selects the subsets of the data that
share the same value for the column “Type 1” and then computes the ranking expression only for those
values. Then, the results for that specific group are projected back to the original rows and Polars
does this for all of the existing groups. The diagram below highlights the ranking computation for
the Pokémon with “Type 1” equal to “Grass”.

<div class="excalidraw">
--8<-- "docs/source/user-guide/expressions/speed_rank_by_type.svg"
</div>

Note how the row for the Pokémon “Golbat” has a “Speed” value of `90`, which is greater than the
value `80` of the Pokémon “Venusaur”, and yet the latter was ranked 1 because “Golbat” and “Venusar”
do not share the same value for the column “Type 1”.

The function `over` accepts an arbitrary number of expressions to specify the groups over which to
perform the computations. We can repeat the ranking above, but over the combination of the columns
“Type 1” and “Type 2” for a more fine-grained ranking:

{{code_block('user-guide/expressions/window','rank-multiple',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:rank-multiple"
```

In general, the results you get with the function `over` can also be achieved with
[an aggregation](aggregation.md) followed by a call to the function `explode`, although the rows
would be in a different order:

{{code_block('user-guide/expressions/window','rank-explode',['explode'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:rank-explode"
```

This shows that, usually, `group_by` and `over` produce results of different shapes:

- `group_by` usually produces a resulting dataframe with as many rows as groups used for
  aggregating; and
- `over` usually produces a dataframe with the same number of rows as the original.

The function `over` does not always produce results with the same number of rows as the original
dataframe, and that is what we explore next.

## Mapping results to dataframe rows

The function `over` accepts a parameter `mapping_strategy` that determines how the results of the
expression over the group are mapped back to the rows of the dataframe.

### `group_to_rows`

The default behaviour is `"group_to_rows"`: the result of the expression over the group should be
the same length as the group and the results are mapped back to the rows of that group.

If the order of the rows is not relevant, the option `"explode"` is more performant. Instead of
mapping the resulting values to the original rows, Polars creates a new dataframe where values from
the same group are next to each other. To help understand the distinction, consider the following
dataframe:

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:athletes"
```

We can sort the athletes by rank within their own countries. If we do so, the Dutch athletes were in
the second, third, and sixth, rows, and they will remain there. What will change is the order of the
names of the athletes, which goes from “B”, “C”, and “F”, to “B”, “F”, and “C”:

{{code_block('user-guide/expressions/window','athletes-sort-over-country',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:athletes-sort-over-country"
```

The diagram below represents this transformation:

<div class="excalidraw">
--8<-- "docs/source/user-guide/expressions/athletes_over_country.svg"
</div>

### `explode`

If we set the parameter `mapping_strategy` to `"explode"`, then athletes of the same country are
grouped together, but the final order of the rows – with respect to the countries – will not be the
same, as the diagram shows:

<div class="excalidraw">
--8<-- "docs/source/user-guide/expressions/athletes_over_country_explode.svg"
</div>

Because Polars does not need to keep track of the positions of the rows of each group, using
`"explode"` is typically faster than `"group_to_rows"`. However, using `"explode"` also requires
more care because it implies reordering the other columns that we wish to keep. The code that
produces this result follows

{{code_block('user-guide/expressions/window','athletes-explode',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:athletes-explode"
```

### `join`

Another possible value for the parameter `mapping_strategy` is `"join"`, which aggregates the
resulting values in a list and repeats the list over all rows of the same group:

{{code_block('user-guide/expressions/window','athletes-join',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:athletes-join"
```

## Windowed aggregation expressions

In case the expression applied to the values of a group produces a scalar value, the scalar is
broadcast across the rows of the group:

{{code_block('user-guide/expressions/window','pokemon-mean',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:pokemon-mean"
```

## More examples

For more exercises, below are some window functions for us to compute:

- sort all Pokémon by type;
- select the first `3` Pokémon per type as `"Type 1"`;
- sort the Pokémon within a type by speed in descending order and select the first `3` as
  `"fastest/group"`;
- sort the Pokémon within a type by attack in descending order and select the first `3` as
  `"strongest/group"`; and
- sort the Pokémon within a type by name and select the first `3` as `"sorted_by_alphabet"`.

{{code_block('user-guide/expressions/window','examples',['over'])}}

```python exec="on" result="text" session="user-guide/window"
--8<-- "python/user-guide/expressions/window.py:examples"
```
