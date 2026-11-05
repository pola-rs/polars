from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.logging import eprint, verbose
from polars._utils.various import qualified_type_name
from polars.exceptions import ComputeError

if TYPE_CHECKING:
    from collections.abc import Callable
    from typing import TypeAlias

    import pyiceberg.catalog

    from polars.io.iceberg._dataset import IcebergCatalogConfig

    IcebergCatalogResolver: TypeAlias = Callable[[str], pyiceberg.catalog.Catalog]

# Set with `pl.Config.set_iceberg_catalog_resolver`; `None` is the default resolver.
# Read at resolve time, so a plan resolves with whichever resolver is current where
# it runs.
ICEBERG_CATALOG_RESOLVER: IcebergCatalogResolver | None = None


def check_catalog_name(name: str) -> str:
    if not name:
        msg = "Iceberg catalog name must not be empty"
        raise ValueError(msg)
    return name


def resolve_catalog(name: str) -> pyiceberg.catalog.Catalog:
    """Turn a catalog name into a PyIceberg catalog, with the current resolver."""
    import pyiceberg.catalog
    from pyiceberg.catalog.noop import NoopCatalog

    resolver = ICEBERG_CATALOG_RESOLVER

    if verbose():
        eprint(
            f"resolve_catalog: {name!r} with "
            f"{'the default' if resolver is None else 'an installed'} resolver"
        )

    if resolver is None:
        try:
            catalog = pyiceberg.catalog.load_catalog(name)
        except Exception as error:
            msg = (
                f"failed to load Iceberg catalog {name!r} ({error = }). "
                "Configure it for PyIceberg (a .pyiceberg.yaml file or "
                "PYICEBERG_CATALOG__* environment variables), or install a resolver "
                "with pl.Config.set_iceberg_catalog_resolver()."
            )
            raise ComputeError(msg) from error
    else:
        # An installed resolver's errors are meant for the user as written; the
        # note says where they come from, as they may surface far from the scan.
        try:
            catalog = resolver(name)
        except Exception as error:
            error.add_note(
                f"raised by the Iceberg catalog resolver for catalog {name!r}"
            )
            raise

    if not isinstance(catalog, pyiceberg.catalog.Catalog):
        msg = (
            f"the Iceberg catalog resolver returned {qualified_type_name(catalog)!r} "
            f"for catalog {name!r}; expected a PyIceberg catalog"
        )
        raise TypeError(msg)

    if isinstance(catalog, NoopCatalog):
        msg = (
            f"the Iceberg catalog resolver returned a NoopCatalog for catalog {name!r}"
        )
        raise TypeError(msg)

    return catalog


def catalog_from(config: IcebergCatalogConfig | str) -> pyiceberg.catalog.Catalog:
    """A new catalog from its config, or by resolving its name."""
    if isinstance(config, str):
        return resolve_catalog(config)
    return config.class_(config.name, **config.properties)
