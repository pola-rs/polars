=========
Selectors
=========

.. currentmodule:: polars

Selectors allow for more intuitive selection of columns from :class:`DataFrame`
or :class:`LazyFrame` objects based on their name, dtype or other properties.
They unify and build on the related functionality that is available through
the :meth:`col` expression and can also broadcast expressions over the selected
columns.

.. note::

    Note that selectors can be negated using the binary "not" operator ``~``, eg:

    .. code-block:: python

        import polars.selectors as s

        # select all columns ending with "_euro"
        s.ends_with("_euro")

        # select all columns NOT ending with "_euro"
        ~s.ends_with("_euro")


Importing
---------

* Selectors are available as functions imported from ``polars.selectors``
* It is recommended to import the module as ``s`` and use from there.

  .. code-block:: python

      import polars.selectors as s
      import polars as pl

      df = pl.DataFrame(
          {
              "w": ["xx", "yy", "xx", "yy", "xx"],
              "x": [1, 2, 1, 4, -2],
              "y": [3.0, 4.5, 1.0, 2.5, -2.0],
              "z": ["a", "b", "a", "b", "b"],
          },
      )
      df.groupby(by=s.string()).agg(s.numeric().sum())


Functions
---------

.. automodule:: polars.selectors
    :members:
    :autosummary:
    :autosummary-no-titles:
