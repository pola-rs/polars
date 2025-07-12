====================
DataType expressions
====================
.. currentmodule:: polars

Data type expressions allow lazily determining a datatype of a column or
expression and using in expressions.

.. autoclass:: DataTypeExpr
    :members:
    :noindex:
    :autosummary:
    :autosummary-nosignatures:

Functions
---------

Available data type expressions functions:

.. autosummary::
   :toctree: api/

   dtype_of
   DataType.to_dtype_expr
