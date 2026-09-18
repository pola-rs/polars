=================
LazyFrameResolver
=================

.. warning::
    This functionality is considered **unstable**. It may be changed
    at any point without it being considered a breaking change.

.. currentmodule:: polars.lazyframe.resolver

Interface for exposing a custom data source to the Polars query engine.

Subclass :class:`LazyFrameResolver` and implement `schema()` and
`resolve_lazyframe()`; the latter is handed the projection, row limit and
filters that the optimizer was able to push down into the source. Call
`lazy()` on the resolver (or `LazyFrame.from_lazyframe_resolver()`) to obtain
a `LazyFrame` that can be used like any other.

.. autosummary::
   :toctree: api/

    LazyFrameResolver
    LazyFrameResolver.lazy
    LazyFrameResolver.schema
    LazyFrameResolver.resolve_lazyframe
    LazyFrameResolver.cse_eq

Resolver arguments and results
------------------------------

.. autosummary::
   :toctree: api/

    FilterExpr
    FilterExpr.pyarrow_expr
    ResolvedLazyFrameProps
