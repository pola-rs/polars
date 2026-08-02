# Trapezoidal integration

A common operation in science and engineering is to compute the area under a curve defined by two
columns. This is often called
[`trapz`](https://numpy.org/doc/stable/reference/generated/numpy.trapz.html) because the
trapezoidal rule is a simple way to approximate the integral.

Given an independent variable `x` and a dependent variable `y`, the trapezoidal rule is:

$$
0.5 \cdot \sum_i (x_{i+1} - x_i) \cdot (y_{i+1} + y_i)
$$

Polars does not provide a dedicated `trapz` method, but the computation can be expressed with
[`shift`](../expressions/window-functions.md) and basic arithmetic:

{{code_block('user-guide/cookbook/trapz','trapz',[])}}

```python exec="on" session="user-guide/cookbook/trapz"
--8<-- "python/user-guide/cookbook/trapz.py:setup"
```

The expression uses `shift` to compare each row with the previous one, which is equivalent to
slicing `x[1:] - x[:-1]` and `y[1:] + y[:-1]`. The first row evaluates to `null` because there is
no previous value, and `sum` ignores `null` values.

## Basic example

{{code_block('user-guide/cookbook/trapz','basic',['DataFrame','select'])}}

```python exec="on" result="text" session="user-guide/cookbook/trapz"
--8<-- "python/user-guide/cookbook/trapz.py:basic"
```

## Per-group integration

The same expression works inside `group_by` when applied with `agg`:

{{code_block('user-guide/cookbook/trapz','grouped',['DataFrame','group_by','agg'])}}

```python exec="on" result="text" session="user-guide/cookbook/trapz"
--8<-- "python/user-guide/cookbook/trapz.py:grouped"
```

!!! note

    The independent variable `x` should be weakly monotonically increasing within each group. If the
    rows are not ordered, sort the data before computing the integral.
