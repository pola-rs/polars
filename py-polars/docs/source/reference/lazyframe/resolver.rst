=================
LazyFrameResolver
=================

.. warning::
    This functionality is considered **unstable**. It may be changed
    at any point without it being considered a breaking change.

.. currentmodule:: polars.lazyframe.resolver

Interface for deferred query building logic.

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
