====
Meta
====

The following methods are available under the `expr.meta` attribute.

These operations allow you to modify the expressions or check for
equality of the expressions themselves instead of answering questions
on the data.

This can for instance be useful to create a set of unique expressions.

  .. code-block:: python

       a1 = pl.col("a")
       a2 = pl.col("a")
       b1 = pl.col("b") + 1
       b2 = pl.col("b") + 2
       b3 = pl.col("b") + 2

       # Create a set of meta expressions to deduplicate
       s = {e.meta for e in [a1, a2, b1, b2, b3]}
       assert len(s) == 3

       # Turn back into expressions to get the final unique expressions
       unique_expressions = [e.as_expression() for e in s]

.. currentmodule:: polars
.. autosummary::
   :toctree: api/
   :template: autosummary/accessor_method.rst

    Expr.meta.as_expression
    Expr.meta.eq
    Expr.meta.has_multiple_outputs
    Expr.meta.is_column
    Expr.meta.is_column_selection
    Expr.meta.is_literal
    Expr.meta.is_regex_projection
    Expr.meta.ne
    Expr.meta.output_name
    Expr.meta.pop
    Expr.meta.root_names
    Expr.meta.serialize
    Expr.meta.show_graph
    Expr.meta.tree_format
    Expr.meta.undo_aliases
    Expr.meta.write_json
