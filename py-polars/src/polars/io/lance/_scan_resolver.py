from __future__ import annotations

from dataclasses import dataclass
from functools import reduce
from time import perf_counter
from typing import TYPE_CHECKING, Any

from polars._utils.logging import eprint
from polars.io.lance._reader import LanceReaderBuilder
from polars.lazyframe.resolver._resolver import (
    LazyFrameResolver,
    ResolvedLazyFrameProps,
)
from polars.schema import Schema

if TYPE_CHECKING:
    import lance

    import polars as pl
    from polars._typing import StorageOptionsDict
    from polars.io.cloud._utils import NoPickleOption
    from polars.io.cloud.credential_provider._builder import CredentialProviderBuilder
    from polars.lazyframe.resolver._resolver import (
        FilterExpr,
    )


@dataclass(kw_only=True)
class LanceScanResolver(LazyFrameResolver):
    dataset_: NoPickleOption[lance.LanceDataset]
    dataset_uri_: str | None

    version: int | str | None
    storage_options: StorageOptionsDict | None
    credential_provider_builder: CredentialProviderBuilder | None

    def schema(self) -> Schema:
        """Fetch the schema of the dataset."""
        return Schema(self.dataset().schema)

    def cse_eq(self, other: object) -> bool:
        return (
            isinstance(other, LanceScanResolver)
            and self.dataset_uri() == other.dataset_uri()
            and self.version == other.version
            and self.storage_options == other.storage_options
            and (
                (
                    self.credential_provider_builder is None
                    and other.credential_provider_builder is None
                )
                or (
                    self.credential_provider_builder is not None
                    and other.credential_provider_builder is not None
                    and self.credential_provider_builder.stable_cache_key()
                    == other.credential_provider_builder.stable_cache_key()
                )
            )
        )

    def resolve_lazyframe(
        self,
        *,
        projection: list[str] | None,
        limit: int | None,
        filters: list[FilterExpr],
        filter_columns: list[str],  # noqa: ARG002
        filter_drop_columns_idx: int | None,
        existing_resolved_version_key: str | None,
    ) -> pl.LazyFrame | tuple[pl.LazyFrame | None, ResolvedLazyFrameProps]:
        """Construct a LazyFrame scan."""
        import polars as pl
        import polars._utils.logging

        verbose = polars._utils.logging.verbose()

        if verbose:
            eprint(
                "LanceScanResolver: resolve_lazyframe(): "
                f"version: {self.version}, "
                f"limit: {limit}, "
                f"projection: {projection}, "
                f"filter_drop_columns_idx: {filter_drop_columns_idx}"
            )

        dataset = self.dataset()
        schema = self.schema()
        projected_schema = (
            {x: schema[x] for x in projection} if projection is not None else schema
        )
        version = self.version if self.version is not None else dataset.version
        version_key = str(version)

        if (
            existing_resolved_version_key is not None
            and existing_resolved_version_key == version_key
        ):
            if verbose:
                eprint(
                    f"LanceScanResolver: resolve_lazyframe(): early return ({version_key = })"
                )

            return None, ResolvedLazyFrameProps(version_key=version_key)

        if verbose:
            eprint("LanceScanResolver: resolve_lazyframe(): dataset.get_fragments()")

        start_time = perf_counter()
        fragments = dataset.get_fragments()
        elapsed = perf_counter() - start_time

        if verbose:
            eprint(
                "LanceScanResolver: resolve_lazyframe(): dataset.get_fragments(): "
                f"{elapsed:.3f}s, "
                f"num_fragments: {len(fragments)}"
            )

        if verbose:
            eprint("LanceScanResolver: serialize fragments")

        start_time = perf_counter()
        fragment_ids = [f"{fragment.fragment_id}" for fragment in fragments]
        elapsed = perf_counter() - start_time

        if verbose:
            eprint(
                f"LanceScanResolver: resolve_lazyframe(): serialize fragments: {elapsed:.3f}s"
            )

        lf = pl.scan_external_reader(
            LanceReaderBuilder(
                dataset=dataset,
                projected_schema=projected_schema,
            ),
            sources=fragment_ids,
            schema=schema,
        )

        if limit is not None:
            lf = lf.head(limit)

        if projection is not None:
            lf = lf.select(projection) if projection else lf.drop("*")

        applied_filters = set()

        if filters:
            lf = lf.filter(reduce(pl.Expr.__and__, (x.expr for x in filters)))
            applied_filters = set(range(len(filters)))

        return lf, ResolvedLazyFrameProps(
            version_key=version_key,
            applied_filters=applied_filters,
        )

    #
    # Accessors
    #

    def dataset_uri(self) -> str:
        """Fetch the dataset URI."""
        if self.dataset_uri_ is None:
            assert self.dataset_.get() is not None
            self.dataset_uri_ = self.dataset().uri

        return self.dataset_uri_

    def dataset(self) -> lance.LanceDataset:
        """Fetch the LanceDataset object."""
        if self.dataset_.get() is None:
            import lance

            from polars.io.cloud.credential_provider._providers import (
                _get_credentials_from_provider_expiry_aware,
            )

            assert self.dataset_uri_ is not None

            credential_provider_creds = {}

            if self.credential_provider_builder and (
                provider := self.credential_provider_builder.build_credential_provider()
            ):
                credential_provider_creds = (
                    _get_credentials_from_provider_expiry_aware(provider) or {}
                )

            dataset = lance.dataset(
                uri=self.dataset_uri_,
                version=self.version,
                storage_options=(
                    {**(self.storage_options or {}), **credential_provider_creds}
                    if self.storage_options is not None
                    or self.credential_provider_builder is not None
                    else None
                ),
            )

            self.dataset_.set(dataset)

        dataset: lance.LanceDataset = self.dataset_.get()  # type: ignore[no-redef]

        if self.version is None:
            dataset.checkout_latest()  # type: ignore[no-untyped-call]
        else:
            dataset = dataset.checkout_version(self.version)

        return dataset

    def __getstate__(self) -> dict[str, Any]:
        self.dataset_uri()
        return self.__dict__

    def __setstate__(self, state: dict[str, Any]) -> None:
        self.__dict__ = state
