=========
Selectors
=========
.. currentmodule:: polars

Selectors allow for more intuitive selection of columns from :class:`DataFrame`
or :class:`LazyFrame` objects based on their name, dtype or other properties.
They unify and build on the related functionality that is available through
the :meth:`col` expression and can also broadcast expressions over the selected
columns.

Importing
---------

* Selectors are available as functions imported from ``polars.selectors``
* Typical/recommended usage is to import the module as ``cs`` and employ selectors from there.

  .. code-block:: python

      import polars.selectors as cs
      import polars as pl

      df = pl.DataFrame(
          {
              "w": ["xx", "yy", "xx", "yy", "xx"],
              "x": [1, 2, 1, 4, -2],
              "y": [3.0, 4.5, 1.0, 2.5, -2.0],
              "z": ["a", "b", "a", "b", "b"],
          },
      )
      df.group_by(by=cs.string()).agg(cs.numeric().sum())

Set operations
--------------

Selectors support ``set`` operations such as:

- UNION:        ``A | B``
- INTERSECTION: ``A & B``
- DIFFERENCE:   ``A - B``
- COMPLEMENT:   ``~A``


Examples
========

.. code-block:: python

    import polars.selectors as cs
    import polars as pl

    # set up an empty dataframe with plenty of columns of various dtypes
    df = pl.DataFrame(
        schema={
            "abc": pl.UInt16,
            "bbb": pl.UInt32,
            "cde": pl.Float64,
            "def": pl.Float32,
            "eee": pl.Boolean,
            "fgg": pl.Boolean,
            "ghi": pl.Time,
            "JJK": pl.Date,
            "Lmn": pl.Duration,
            "opp": pl.Datetime("ms"),
            "qqR": pl.Utf8,
        },
    )

    # Select the UNION of temporal, strings and columns that start with "e"
    assert df.select(cs.temporal() | cs.string() | cs.starts_with("e")).schema == {
        "eee": pl.Boolean,
        "ghi": pl.Time,
        "JJK": pl.Date,
        "Lmn": pl.Duration,
        "opp": pl.Datetime("ms"),
        "qqR": pl.Utf8,
    }

    # Select the INTERSECTION of temporal and column names that match "opp" OR "JJK"
    assert df.select(cs.temporal() & cs.matches("opp|JJK")).schema == {
        "JJK": pl.Date,
        "opp": pl.Datetime("ms"),
    }

    # Select the DIFFERENCE of temporal columns and columns that contain the name "opp" OR "JJK"
    assert df.select(cs.temporal() - cs.matches("opp|JJK")).schema == {
        "ghi": pl.Time,
        "Lmn": pl.Duration,
    }

    # Select the COMPLEMENT of all columns of dtypes Duration and Time
    assert df.select(~cs.by_dtype([pl.Duration, pl.Time])).schema == {
        "abc": pl.UInt16,
        "bbb": pl.UInt32,
        "cde": pl.Float64,
        "def": pl.Float32,
        "eee": pl.Boolean,
        "fgg": pl.Boolean,
        "JJK": pl.Date,
        "opp": pl.Datetime("ms"),
        "qqR": pl.Utf8,
    }


.. note::

    If you don't want to use the set operations on the selectors, you can materialize them as ``expressions``
    by calling ``as_expr``. This ensures the operations ``OR, AND, etc`` are dispatched to the underlying
    expressions instead.

Functions
---------

Available selector functions:

.. automodule:: polars.selectors
    :members:
    :autosummary:
    :autosummary-no-titles:
