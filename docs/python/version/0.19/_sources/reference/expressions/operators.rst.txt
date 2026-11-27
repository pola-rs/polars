=========
Operators
=========

Polars supports native Python operators for all common operations;
these operators are also available as methods on the :class:`Expr`
class.

Conjunction
~~~~~~~~~~~

.. currentmodule:: polars
.. autosummary::
   :toctree: api/

    Expr.and_
    Expr.or_

Comparison
~~~~~~~~~~

.. autosummary::
   :toctree: api/

    Expr.eq
    Expr.eq_missing
    Expr.ge
    Expr.gt
    Expr.le
    Expr.lt
    Expr.ne
    Expr.ne_missing

Numeric
~~~~~~~

.. autosummary::
   :toctree: api/

    Expr.add
    Expr.floordiv
    Expr.mod
    Expr.mul
    Expr.sub
    Expr.truediv
    Expr.pow


Binary
~~~~~~

.. autosummary::
   :toctree: api/

    Expr.xor
