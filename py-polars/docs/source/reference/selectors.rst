=========
Selectors
=========

.. currentmodule:: polars

The selector class :class:`s` allows for the intuitive selection of
:class:`DataFrame` or :class:`LazyFrame` columns based on their name,
dtype or other properties. This class unifies and builds on similar
functionality that is available through the :meth:`col` expression.

.. note::

    Note that selectors can be negated using the binary "not" operator ``~``, eg:

    .. code-block:: python

        df.select(~s.ends_with("_euro"))


=========
Methods
=========

Selectors are available as static methods on :class:`s`:

.. currentmodule:: polars.selectors

.. py:class:: s
    :canonical: polars.selectors.s

     .. automethod:: all
     .. automethod:: by_dtype
     .. automethod:: by_name
     .. automethod:: contains
     .. automethod:: ends_with
     .. automethod:: first
     .. automethod:: float
     .. automethod:: integer
     .. automethod:: last
     .. automethod:: matches
     .. automethod:: numeric
     .. automethod:: starts_with
     .. automethod:: temporal
