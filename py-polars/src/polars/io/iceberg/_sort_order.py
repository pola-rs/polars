"""The sort order of `ALTER TABLE ... SET | RESET SORTED BY` (Iceberg)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyiceberg.table
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.sorting import SortOrder


def sort_order_change(
    metadata: TableMetadata,
    fields: list[tuple[list[str], str, bool, bool]],
    name: str,
) -> SortOrder | int | None:
    """
    The change that makes `fields` the default sort order of table `name`.

    Each field is (source path, transform, descending, nulls first). Returns a new sort
    order, the id of an existing one, or `None` if the default sort order has them.
    """
    from pyiceberg.table.sorting import (
        UNSORTED_SORT_ORDER,
        NullOrder,
        SortDirection,
        SortField,
        SortOrder,
    )
    from pyiceberg.transforms import parse_transform

    from polars.io.iceberg._partition_evolution import partition_source

    schema = metadata.schema()
    target = []
    for path, transform_name, descending, nulls_first in fields:
        transform = parse_transform(transform_name)
        source = partition_source(schema, path, transform, name, kind="sort")
        target.append(
            SortField(
                source_id=source.field_id,
                transform=transform,
                direction=SortDirection.DESC if descending else SortDirection.ASC,
                # explicit: SortField's default null order depends on the direction
                null_order=NullOrder.NULLS_FIRST
                if nulls_first
                else NullOrder.NULLS_LAST,
            )
        )

    # An existing order with these fields is reused, as Iceberg does.
    current = metadata.sort_order_by_id(metadata.default_sort_order_id)
    if current is not None and current.fields == target:
        return None
    for order in metadata.sort_orders:
        if order.fields == target:
            return order.order_id
    if not target:
        return UNSORTED_SORT_ORDER
    order_id = max((order.order_id for order in metadata.sort_orders), default=0) + 1
    return SortOrder(*target, order_id=order_id)


def commit_sort_order_change(
    table: pyiceberg.table.Table,
    metadata: TableMetadata,
    change: SortOrder | int,
) -> None:
    """Commit `change` against `metadata`, failing if the table changed since."""
    from pyiceberg.table.sorting import SortOrder
    from pyiceberg.table.update import (
        AddSortOrderUpdate,
        AssertCurrentSchemaId,
        AssertDefaultSortOrderId,
        SetDefaultSortOrderUpdate,
    )

    updates: tuple[Any, ...]
    if isinstance(change, SortOrder):
        # The unsorted order is always 0. Else -1: the order just added, whichever id
        # the catalog gives it (PyIceberg takes -1 as the highest id, which it is).
        default = 0 if change.is_unsorted else -1
        updates = (
            AddSortOrderUpdate(sort_order=change),  # type: ignore[call-arg]
            SetDefaultSortOrderUpdate(sort_order_id=default),  # type: ignore[call-arg]
        )
    else:
        updates = (SetDefaultSortOrderUpdate(sort_order_id=change),)  # type: ignore[call-arg]
    requirements = (
        AssertDefaultSortOrderId(default_sort_order_id=metadata.default_sort_order_id),  # type: ignore[call-arg]
        AssertCurrentSchemaId(current_schema_id=metadata.current_schema_id),  # type: ignore[call-arg]
    )
    tx = table.transaction()
    tx._stage(updates, requirements)
    tx.commit_transaction()
