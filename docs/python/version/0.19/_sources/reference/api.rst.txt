=================
Extending the API
=================
.. currentmodule:: polars

Providing new functionality
---------------------------

These functions allow you to register custom functionality in a dedicated
namespace on the underlying polars classes without requiring subclassing
or mixins. Expr, DataFrame, LazyFrame, and Series are all supported targets.

This feature is primarily intended for use by library authors providing
domain-specific capabilities which may not exist (or belong) in the
core library.


Available registrations
-----------------------

.. currentmodule:: polars.api
.. autosummary::
   :toctree: api/

    register_expr_namespace
    register_dataframe_namespace
    register_lazyframe_namespace
    register_series_namespace

.. note::

   You cannot override existing polars namespaces (such as ``.str`` or ``.dt``), and attempting to do so
   will raise an `AttributeError <https://docs.python.org/3/library/exceptions.html#AttributeError>`_.
   However, you *can* override other custom namespaces (which will only generate a
   `UserWarning <https://docs.python.org/3/library/exceptions.html#UserWarning>`_).


Examples
--------

.. tab-set::

    .. tab-item:: Expr

        .. code-block:: python

            @pl.api.register_expr_namespace("greetings")
            class Greetings:
                def __init__(self, expr: pl.Expr):
                    self._expr = expr

                def hello(self) -> pl.Expr:
                    return (pl.lit("Hello ") + self._expr).alias("hi there")

                def goodbye(self) -> pl.Expr:
                    return (pl.lit("Sayōnara ") + self._expr).alias("bye")


            pl.DataFrame(data=["world", "world!", "world!!"]).select(
                [
                    pl.all().greetings.hello(),
                    pl.all().greetings.goodbye(),
                ]
            )

            # shape: (3, 1)   shape: (3, 2)
            # ┌──────────┐    ┌───────────────┬──────────────────┐
            # │ column_0 │    │ hi there      ┆ bye              │
            # │ ---      │    │ ---           ┆ ---              │
            # │ str      │    │ str           ┆ str              │
            # ╞══════════╡ >> ╞═══════════════╪══════════════════╡
            # │ world    │    │ Hello world   ┆ Sayōnara world   │
            # │ world!   │    │ Hello world!  ┆ Sayōnara world!  │
            # │ world!!  │    │ Hello world!! ┆ Sayōnara world!! │
            # └──────────┘    └───────────────┴──────────────────┘

    .. tab-item:: DataFrame

        .. code-block:: python

            @pl.api.register_dataframe_namespace("split")
            class SplitFrame:
                def __init__(self, df: pl.DataFrame):
                    self._df = df

                def by_alternate_rows(self) -> list[pl.DataFrame]:
                    df = self._df.with_row_count(name="n")
                    return [
                        df.filter((pl.col("n") % 2) == 0).drop("n"),
                        df.filter((pl.col("n") % 2) != 0).drop("n"),
                    ]


            pl.DataFrame(
                data=["aaa", "bbb", "ccc", "ddd", "eee", "fff"],
                columns=[("txt", pl.Utf8)],
            ).split.by_alternate_rows()

            # [┌─────┐  ┌─────┐
            #  │ txt │  │ txt │
            #  │ --- │  │ --- │
            #  │ str │  │ str │
            #  ╞═════╡  ╞═════╡
            #  │ aaa │  │ bbb │
            #  │ ccc │  │ ddd │
            #  │ eee │  │ fff │
            #  └─────┘, └─────┘]

    .. tab-item:: LazyFrame

        .. code-block:: python

            @pl.api.register_lazyframe_namespace("types")
            class DTypeOperations:
                def __init__(self, ldf: pl.LazyFrame):
                    self._ldf = ldf

                def upcast_integer_types(self) -> pl.LazyFrame:
                    return self._ldf.with_columns(
                        pl.col(tp).cast(pl.Int64)
                        for tp in (pl.Int8, pl.Int16, pl.Int32)
                    )


            ldf = pl.DataFrame(
                data={"a": [1, 2], "b": [3, 4], "c": [5.6, 6.7]},
                columns=[("a", pl.Int16), ("b", pl.Int32), ("c", pl.Float32)],
            ).lazy()

            ldf.types.upcast_integer_types()

            # shape: (2, 3)          shape: (2, 3)
            # ┌─────┬─────┬─────┐    ┌─────┬─────┬─────┐
            # │ a   ┆ b   ┆ c   │    │ a   ┆ b   ┆ c   │
            # │ --- ┆ --- ┆ --- │    │ --- ┆ --- ┆ --- │
            # │ i16 ┆ i32 ┆ f32 │ >> │ i64 ┆ i64 ┆ f32 │
            # ╞═════╪═════╪═════╡    ╞═════╪═════╪═════╡
            # │ 1   ┆ 3   ┆ 5.6 │    │ 1   ┆ 3   ┆ 5.6 │
            # │ 2   ┆ 4   ┆ 6.7 │    │ 2   ┆ 4   ┆ 6.7 │
            # └─────┴─────┴─────┘    └─────┴─────┴─────┘

    .. tab-item:: Series

        .. code-block:: python

            @pl.api.register_series_namespace("math")
            class MathShortcuts:
                def __init__(self, s: pl.Series):
                    self._s = s

                def square(self) -> pl.Series:
                    return self._s * self._s

                def cube(self) -> pl.Series:
                    return self._s * self._s * self._s


            s = pl.Series("n", [1, 2, 3, 4, 5])

            s2 = s.math.square().rename("n2", in_place=True)
            s3 = s.math.cube().rename("n3", in_place=True)

            # shape: (5,)          shape: (5,)           shape: (5,)
            # Series: 'n' [i64]    Series: 'n2' [i64]    Series: 'n3' [i64]
            # [                    [                     [
            #     1                    1                      1
            #     2                    4                      8
            #     3                    9                      27
            #     4                    16                     64
            #     5                    25                    125
            # ]                    ]                    ]
