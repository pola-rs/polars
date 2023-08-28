# Basic Operators

This section describes how to use basic operators (e.g. addition, substraction) in conjunction with Expressions. We will provide various examples using different themes in the context of the following dataframe.

!!! note Operator Overloading

    In Rust and Python it is possible to use the operators directly (as in `+ - * / < > `) as the language allows operator overloading. For instance, the operator `+` translates to the `.add()` method. In NodeJS this is not possible and you must use the methods themselves, in python and rust you can choose which one you prefer.

{{code_block('user-guide/expressions/operators','dataframe',['DataFrame'])}}

```python exec="on" result="text" session="user-guide/operators"
--8<-- "python/user-guide/expressions/operators.py:setup"
--8<-- "python/user-guide/expressions/operators.py:dataframe"
```

### Numerical

{{code_block('user-guide/expressions/operators','numerical',['operators'])}}

```python exec="on" result="text" session="user-guide/operators"
--8<-- "python/user-guide/expressions/operators.py:numerical"
```

### Logical

{{code_block('user-guide/expressions/operators','logical',['operators'])}}

```python exec="on" result="text" session="user-guide/operators"
--8<-- "python/user-guide/expressions/operators.py:logical"
```