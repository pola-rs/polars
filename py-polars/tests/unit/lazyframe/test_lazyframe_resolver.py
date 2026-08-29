from __future__ import annotations

import pickle
from functools import reduce
from itertools import chain, cycle
from typing import TYPE_CHECKING, Any
from unittest.mock import Mock

import pytest

import polars as pl
from polars._plr import ComputeError
from polars.exceptions import ShapeError
from polars.lazyframe.resolver import LazyFrameResolver
from polars.lazyframe.resolver._resolver import ResolvedLazyFrameProps
from polars.testing.asserts.frame import assert_frame_equal

if TYPE_CHECKING:
    from collections.abc import Callable

    from polars._typing import SchemaDict
    from polars.lazyframe.resolver._resolver import FilterExpr


class InMemoryLazyFrameResolver(LazyFrameResolver):  # noqa: D101
    def __init__(
        self,
        lf: pl.LazyFrame,
        *,
        cse_eq: Callable[
            [InMemoryLazyFrameResolver, InMemoryLazyFrameResolver], bool
        ] = LazyFrameResolver.cse_eq,
        use_filter_drop_columns_idx: bool = False,
    ) -> None:
        self.lf = lf
        self.cse_eq_f = cse_eq
        self.use_filter_drop_columns_idx = use_filter_drop_columns_idx

    def schema(self) -> SchemaDict:
        return self.lf.collect_schema()

    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],
        filter_columns: list[str],
        filter_drop_columns_idx: int | None,
        existing_resolved_version_key: str | None,
    ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        lf = self.lf

        if limit is not None:
            lf = lf.head(limit)

        if projection is not None:
            lf = lf.select(projection) if projection else lf.drop("*")

        applied_filters = set()

        if filters:
            lf = lf.filter(reduce(pl.Expr.__and__, (x.expr for x in filters)))
            applied_filters = set(range(len(filters)))

            if self.use_filter_drop_columns_idx and filter_drop_columns_idx is not None:
                assert projection is not None
                names = projection[:filter_drop_columns_idx]

                lf = lf.select(names) if names else lf.drop("*")

        if self.use_filter_drop_columns_idx and applied_filters:
            return lf, ResolvedLazyFrameProps(applied_filters=applied_filters)

        return lf

    def cse_eq(self, other: InMemoryLazyFrameResolver) -> bool:
        return self.cse_eq_f(self, other)


def test_lazyframe_resolver() -> None:
    resolver = InMemoryLazyFrameResolver(
        pl.LazyFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]})
    )
    resolver.resolve_lazyframe = Mock(wraps=resolver.resolve_lazyframe)  # type: ignore[method-assign]

    lf = resolver.lazy()

    assert_frame_equal(
        lf.collect(),
        pl.DataFrame({"a": [0, 1, 2], "b": ["a", "b", "c"]}),
    )

    q = lf.filter(pl.col("a") == 0)
    assert_frame_equal(
        q.collect(),
        pl.DataFrame({"a": [0], "b": ["a"]}),
    )

    q = lf.filter(pl.col("a") == 0).drop("*")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(height=1),
    )

    assert resolver.resolve_lazyframe.call_args.kwargs["projection"] == ["a"]
    assert resolver.resolve_lazyframe.call_args.kwargs["filter_drop_columns_idx"] == 0

    q = lf.slice(1, 1).drop("*")
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(height=1),
    )

    q = lf.filter(pl.col("a") == 0).select(pl.len())

    assert q.collect().item() == 1


def test_lazyframe_resolver_filter() -> None:
    lf = pl.LazyFrame(
        {
            "a": [0, 1],
            "mask": [True, False],
        }
    )
    resolver = InMemoryLazyFrameResolver(lf, use_filter_drop_columns_idx=True)
    resolver.resolve_lazyframe = Mock(wraps=resolver.resolve_lazyframe)  # type: ignore[method-assign]

    q = resolver.lazy().filter("mask").drop("*")
    plan = q.explain()

    assert "simple π 0/0 []" in plan[plan.index("RESOLVER") :]
    assert_frame_equal(
        q.collect(),
        pl.DataFrame(height=1),
    )


def test_lazyframe_resolver_cse() -> None:
    resolver = InMemoryLazyFrameResolver(pl.LazyFrame({"a": [1]}))
    resolver.resolve_lazyframe = Mock(wraps=resolver.resolve_lazyframe)  # type: ignore[method-assign]

    lf = resolver.lazy()
    q = pl.concat([lf, lf])
    plan = q.explain()

    assert "CACHE" not in plan
    assert_frame_equal(q.collect(), pl.DataFrame({"a": [1, 1]}))
    assert resolver.resolve_lazyframe.call_count == 2

    def object_id_eq(self: LazyFrameResolver, other: LazyFrameResolver) -> bool:
        return self is other

    resolver = InMemoryLazyFrameResolver(pl.LazyFrame({"a": [1]}), cse_eq=object_id_eq)
    resolver.resolve_lazyframe = Mock(wraps=resolver.resolve_lazyframe)  # type: ignore[method-assign]

    lf = resolver.lazy()
    q = pl.concat([lf, lf])
    plan = q.explain()

    assert plan.count("CACHE") == 2
    assert_frame_equal(q.collect(), pl.DataFrame({"a": [1, 1]}))
    assert resolver.resolve_lazyframe.call_count == 1


def test_lazyframe_resolver_cse_after_pickle() -> None:
    def values_eq(
        self: InMemoryLazyFrameResolver,
        other: InMemoryLazyFrameResolver,
    ) -> bool:
        return pl.DataFrame.equals(self.lf.collect(), other.lf.collect())

    resolver = InMemoryLazyFrameResolver(pl.LazyFrame({"a": [1]}), cse_eq=values_eq)
    lf = resolver.lazy()
    q = pl.concat(
        [
            pickle.loads(pickle.dumps(lf)),
            pickle.loads(pickle.dumps(lf)),
        ]
    )
    plan = q.explain()

    assert plan.count("CACHE") == 2
    assert_frame_equal(q.collect(), pl.DataFrame({"a": [1, 1]}))


def test_lazyframe_resolver_versioned_caching() -> None:
    resolver = InMemoryLazyFrameResolver(pl.LazyFrame({"a": [1]}))
    resolver.resolve_lazyframe = Mock(wraps=resolver.resolve_lazyframe)  # type: ignore[method-assign]

    lf = resolver.lazy()

    # Unversioned return is not re-called
    assert_frame_equal(lf.collect(), pl.DataFrame({"a": [1]}))
    assert_frame_equal(lf.collect(), pl.DataFrame({"a": [1]}))
    assert resolver.resolve_lazyframe.call_count == 1

    class VersionedResolver(InMemoryLazyFrameResolver):
        def __init__(
            self,
            lf: pl.LazyFrame,
            *,
            cse_eq: Callable[
                [InMemoryLazyFrameResolver, InMemoryLazyFrameResolver], bool
            ] = LazyFrameResolver.cse_eq,
        ) -> None:
            super().__init__(lf, cse_eq=cse_eq)
            self.version_keys_iter = chain([1, 2], cycle([3]))
            self.last_resolved_lazyframe: (
                tuple[pl.LazyFrame | None, ResolvedLazyFrameProps] | None
            ) = None

        def resolve_lazyframe(
            self,
            *,
            projection: list[str] | None,
            limit: int | None,
            filters: list[FilterExpr],
            filter_columns: list[str],
            filter_drop_columns_idx: int | None,
            existing_resolved_version_key: str | None,
        ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
            version_key = next(self.version_keys_iter)

            self.lf = self.lf.with_columns(version=version_key)
            lf = None

            if existing_resolved_version_key != str(version_key):
                lf = super().resolve_lazyframe(
                    projection=projection,
                    limit=limit,
                    filters=filters,
                    filter_columns=filter_columns,
                    filter_drop_columns_idx=filter_drop_columns_idx,
                    existing_resolved_version_key=existing_resolved_version_key,
                )

                assert isinstance(lf, pl.LazyFrame)

            ret = lf, ResolvedLazyFrameProps(version_key=str(version_key))

            self.last_resolved_lazyframe = ret  # type: ignore[assignment]

            return ret  # type: ignore[return-value]

    resolver = VersionedResolver(pl.LazyFrame({"version": [-1]}))
    lf = resolver.lazy()

    def last_resolved() -> tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        assert resolver.last_resolved_lazyframe is not None
        return resolver.last_resolved_lazyframe

    assert lf.collect().item() == 1
    assert isinstance(last_resolved()[0], pl.LazyFrame)

    assert lf.collect().item() == 2
    assert isinstance(last_resolved()[0], pl.LazyFrame)

    assert lf.collect().item() == 3
    assert isinstance(last_resolved()[0], pl.LazyFrame)

    assert lf.collect().item() == 3
    assert last_resolved()[0] is None

    # Change in resolve parameters requires re-resolve.
    assert lf.drop("version").collect().shape == (1, 0)
    assert isinstance(last_resolved()[0], pl.LazyFrame)


def test_lazyframe_resolver_applied_filters() -> None:
    class FilterOverrideResolver(InMemoryLazyFrameResolver):
        def __init__(
            self,
            lf: pl.LazyFrame,
            *,
            applied_filters: set[int],
            cse_eq: Callable[
                [InMemoryLazyFrameResolver, InMemoryLazyFrameResolver], bool
            ] = LazyFrameResolver.cse_eq,
        ) -> None:
            super().__init__(lf, cse_eq=cse_eq)
            self.applied_filters = applied_filters

        def resolve_lazyframe(
            self,
            *,
            projection: list[str] | None,
            limit: int | None,
            filters: list[FilterExpr],
            filter_columns: list[str],
            filter_drop_columns_idx: int | None,
            existing_resolved_version_key: str | None,
        ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
            del filters

            lf = super().resolve_lazyframe(
                projection=projection,
                limit=limit,
                filters=[],
                filter_columns=[],
                filter_drop_columns_idx=filter_drop_columns_idx,
                existing_resolved_version_key=existing_resolved_version_key,
            )

            assert isinstance(lf, pl.LazyFrame)

            return lf, ResolvedLazyFrameProps(applied_filters=self.applied_filters)

    q = (
        FilterOverrideResolver(pl.LazyFrame({"a": 1}), applied_filters=set())
        .lazy()
        .filter(pl.col("a") != 1)
        .lazy()
    )
    assert_frame_equal(q.collect(), pl.DataFrame(schema={"a": pl.Int64}))

    q = (
        FilterOverrideResolver(pl.LazyFrame({"a": 1}), applied_filters={0})
        .lazy()
        .filter(pl.col("a") != 1)
    )

    assert_frame_equal(q.collect(), pl.DataFrame({"a": 1}))

    with pytest.raises(
        ShapeError,
        match=r"index \(i = 1\) contained in `applied_filters` out of bounds for n_filters = 1",
    ):
        (
            FilterOverrideResolver(pl.LazyFrame({"a": 1}), applied_filters={1})
            .lazy()
            .filter(pl.col("a") != 1)
            .collect()
        )


def test_lazyframe_resolver_nesting() -> None:

    class NestedResolver(LazyFrameResolver):
        def __init__(self, depth: int, level_width: int) -> None:
            self.depth = depth
            self.level_width = level_width

        def schema(self) -> SchemaDict:
            return {"x": pl.Int64}

        def resolve_lazyframe(
            self,
            *,
            projection: list[str] | None,
            limit: int | None,
            filters: list[FilterExpr],
            filter_columns: list[str],
            filter_drop_columns_idx: int | None,
            existing_resolved_version_key: str | None,
        ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
            if self.depth <= 0:
                return pl.LazyFrame({"x": 0})

            return pl.concat(
                [
                    NestedResolver(self.depth - 1, self.level_width).lazy()
                    for _ in range(self.level_width)
                ]
            )

    q = NestedResolver(200, 1).lazy()
    assert q.collect().item() == 0

    q = NestedResolver(1, 200).lazy()
    assert_frame_equal(
        q.collect().group_by("x").agg(pl.len()),
        pl.DataFrame(
            {"x": 0, "len": 200},
            schema_overrides={"len": pl.get_index_type()},
        ),
    )


def test_lazyframe_versioned_resolver_incorrect_returns() -> None:
    class Resolver(LazyFrameResolver):
        def schema(self) -> SchemaDict:
            return {}

        def resolve_lazyframe(
            self, **kwargs: Any
        ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
            raise NotImplementedError

    resolver = Resolver()
    q = resolver.lazy()

    resolver.resolve_lazyframe = lambda **kw: (  # type: ignore[method-assign]
        pl.LazyFrame(height=3),
        ResolvedLazyFrameProps(version_key="1"),
    )

    assert q.collect().shape == (3, 0)

    resolver.resolve_lazyframe = lambda **kw: (  # type: ignore[method-assign]
        None,
        ResolvedLazyFrameProps(version_key="2"),
    )

    with pytest.raises(
        ComputeError,
        match=r"returned None.*version key does not match or was not found",
    ):
        q.collect()

    resolver = Resolver()
    q = resolver.lazy()

    resolver.resolve_lazyframe = lambda **kw: (  # type: ignore[method-assign]
        None,
        ResolvedLazyFrameProps(version_key="1"),
    )

    with pytest.raises(
        ComputeError,
        match=r"returned None.*version key does not match or was not found",
    ):
        q.collect()

    resolver.resolve_lazyframe = lambda **kw: (  # type: ignore[method-assign]
        pl.LazyFrame(height=3),
        ResolvedLazyFrameProps(version_key="1"),
    )

    assert q.collect().shape == (3, 0)

    resolver.resolve_lazyframe = lambda **kw: (  # type: ignore[method-assign]
        pl.LazyFrame(height=1),
        ResolvedLazyFrameProps(version_key="1"),
    )

    assert q.collect().shape == (1, 0)

    resolver.resolve_lazyframe = lambda **kw: (  # type: ignore[method-assign]
        None,
        ResolvedLazyFrameProps(version_key="1"),
    )

    assert q.collect().shape == (1, 0)
