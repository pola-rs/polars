"""The metadata changes of `CREATE OR REPLACE TABLE` on an Iceberg table."""

from __future__ import annotations

from typing import TYPE_CHECKING

from polars.exceptions import DuplicateError, InvalidOperationError

if TYPE_CHECKING:
    import pyiceberg.partitioning
    from pyiceberg.schema import Schema
    from pyiceberg.table.metadata import TableMetadata
    from pyiceberg.table.name_mapping import MappedField
    from pyiceberg.table.update import TableRequirement, TableUpdate
    from pyiceberg.types import IcebergType, NestedField

_DROP_CREATE = "use DROP TABLE and CREATE TABLE instead"


def replace_changes(
    metadata: TableMetadata,
    schema: Schema,
    partition_by: list[tuple[list[str], str]],
    properties: dict[str, str],
    location: str | None,
    name: str,
) -> tuple[tuple[TableUpdate, ...], tuple[TableRequirement, ...]]:
    """
    The updates and requirements that replace table `name` by a new one.

    The new table has `schema`, the partitioning `partition_by` and no sort order. Its
    properties are the table's with `properties` set. The updates remove `main`: the
    snapshot the caller adds has no parent. The requirements fail the commit if the
    table changed since `metadata` was loaded.
    """
    from pyiceberg.table import TableProperties
    from pyiceberg.table.update import (
        AddPartitionSpecUpdate,
        AddSchemaUpdate,
        AddSortOrderUpdate,
        AssertCurrentSchemaId,
        AssertDefaultSortOrderId,
        AssertDefaultSpecId,
        AssertLastAssignedFieldId,
        AssertLastAssignedPartitionId,
        AssertRefSnapshotId,
        RemoveSnapshotRefUpdate,
        SetCurrentSchemaUpdate,
        SetDefaultSortOrderUpdate,
        SetDefaultSpecUpdate,
        SetPropertiesUpdate,
    )

    from polars.io.iceberg._sort_order import sort_order_change

    properties = _checked_properties(metadata, properties, location, name)
    updates: list[TableUpdate] = []
    main = metadata.refs.get("main")
    if main is not None:
        updates.append(RemoveSnapshotRefUpdate(ref_name="main"))  # type: ignore[call-arg]

    new_schema = replacement_schema(metadata, schema)
    _check_partition_names(metadata, new_schema, name)
    if new_schema.schema_id != metadata.current_schema_id:
        if any(s.schema_id == new_schema.schema_id for s in metadata.schemas):
            updates.append(SetCurrentSchemaUpdate(schema_id=new_schema.schema_id))  # type: ignore[call-arg]
        else:
            # -1: the schema just added, whichever id the catalog gives it
            updates += [
                AddSchemaUpdate(schema=new_schema),  # type: ignore[call-arg]
                SetCurrentSchemaUpdate(schema_id=-1),  # type: ignore[call-arg]
            ]
    # the new schema, with the current spec and sort order: PyIceberg's metadata
    # update would check those against it (the updates are applied together)
    schemas = list(metadata.schemas)
    if all(s.schema_id != new_schema.schema_id for s in schemas):
        schemas.append(new_schema)
    staged = metadata.model_copy(
        update={
            "schemas": schemas,
            "current_schema_id": new_schema.schema_id,
            "last_column_id": max(metadata.last_column_id, new_schema.highest_field_id),
        }
    )

    spec = _replacement_spec(staged, partition_by, name)
    if isinstance(spec, int):
        updates.append(SetDefaultSpecUpdate(spec_id=spec))  # type: ignore[call-arg]
    elif spec is not None:
        updates += [
            AddPartitionSpecUpdate(spec=spec),  # type: ignore[call-arg]
            SetDefaultSpecUpdate(spec_id=-1),  # type: ignore[call-arg]
        ]

    order = sort_order_change(staged, [], name)
    if isinstance(order, int):
        updates.append(SetDefaultSortOrderUpdate(sort_order_id=order))  # type: ignore[call-arg]
    elif order is not None:
        # the unsorted order is always 0
        updates += [
            AddSortOrderUpdate(sort_order=order),  # type: ignore[call-arg]
            SetDefaultSortOrderUpdate(sort_order_id=0),  # type: ignore[call-arg]
        ]

    key = TableProperties.DEFAULT_NAME_MAPPING
    mappings = [metadata.properties.get(key), properties.get(key)]
    if any(mapping is not None for mapping in mappings):
        # files without field ids are read by the mapping: it keeps its old names
        properties[key] = _replacement_mapping(mappings, new_schema, name)
    if properties:
        updates.append(SetPropertiesUpdate(updates=properties))  # type: ignore[call-arg]
    if not updates:
        # an update, so that the catalog checks the requirements
        updates.append(
            SetCurrentSchemaUpdate(schema_id=metadata.current_schema_id)  # type: ignore[call-arg]
        )

    requirements = (
        AssertRefSnapshotId(  # type: ignore[call-arg]
            ref="main", snapshot_id=None if main is None else main.snapshot_id
        ),
        AssertCurrentSchemaId(current_schema_id=metadata.current_schema_id),  # type: ignore[call-arg]
        AssertDefaultSpecId(default_spec_id=metadata.default_spec_id),  # type: ignore[call-arg]
        AssertDefaultSortOrderId(  # type: ignore[call-arg]
            default_sort_order_id=metadata.default_sort_order_id
        ),
        AssertLastAssignedFieldId(last_assigned_field_id=metadata.last_column_id),  # type: ignore[call-arg]
        AssertLastAssignedPartitionId(  # type: ignore[call-arg]
            last_assigned_partition_id=metadata.last_partition_id
        ),
    )
    return tuple(updates), requirements


def _checked_properties(
    metadata: TableMetadata,
    properties: dict[str, str],
    location: str | None,
    name: str,
) -> dict[str, str]:
    """`properties` to set, without an unchanged format version; raise on a change."""
    if metadata.format_version > 2:
        msg = (
            f"replacing an Iceberg v{metadata.format_version} table is not supported "
            f"yet: table '{name}'; {_DROP_CREATE}"
        )
        raise NotImplementedError(msg)
    if location is not None and location.rstrip("/") != metadata.location.rstrip("/"):
        msg = (
            f"table '{name}' cannot be replaced at another location ('{location}', "
            f"table location '{metadata.location}'); {_DROP_CREATE}"
        )
        raise InvalidOperationError(msg)
    properties = dict(properties)
    version = properties.pop("format-version", None)
    if version is not None and version != str(metadata.format_version):
        msg = (
            f"the format version of table '{name}' (v{metadata.format_version}) cannot "
            f"change in a replace; {_DROP_CREATE}"
        )
        raise InvalidOperationError(msg)
    return properties


def _check_partition_names(metadata: TableMetadata, schema: Schema, name: str) -> None:
    """
    Raise if a column of `schema` takes the name of another column's partition field.

    Iceberg keeps every partition spec. A column with the name of a partition field of
    another source fails the commit on some catalogs, or later schema changes in
    PyIceberg (as when an identity partition column gets a new field id).
    """
    from pyiceberg.partitioning import validate_partition_name

    for spec in metadata.partition_specs:
        for field in spec.fields:
            try:
                validate_partition_name(
                    field.name, field.transform, field.source_id, schema, set()
                )
            except ValueError as error:
                msg = (
                    f"column '{field.name}' of table '{name}' would have the name of a "
                    f"partition field of partition spec {spec.spec_id}, which partitions "
                    "another column (as when a partition column gets a new type): give "
                    f"the column another name, or {_DROP_CREATE} (which drops the "
                    "table's history)"
                )
                raise InvalidOperationError(msg) from error


def replacement_schema(metadata: TableMetadata, schema: Schema) -> Schema:
    """
    `schema` with the field ids of the current schema of `metadata`.

    A field keeps the id of the current field of its full name if that one can evolve
    into it: the same type or a type promotion, or the same kind of nested type. Other
    fields get new ids. A retained schema equal to the result is returned as is.
    """
    from pyiceberg.schema import Schema
    from pyiceberg.types import ListType, MapType, NestedField, StructType

    next_id = metadata.last_column_id

    def new_id() -> int:
        nonlocal next_id
        next_id += 1
        return next_id

    def struct(new: StructType, old: StructType | None) -> StructType:
        return StructType(
            *(
                field(f, None if old is None else old.field_by_name(f.name))
                for f in new.fields
            )
        )

    def field(new: NestedField, old: NestedField | None) -> NestedField:
        keep = old is not None and _evolves(old.field_type, new.field_type)
        field_id = old.field_id if keep and old is not None else new_id()
        return NestedField(
            field_id=field_id,
            name=new.name,
            field_type=typed(new.field_type, old.field_type if keep and old else None),
            required=new.required,
            doc=new.doc,
        )

    def typed(new: IcebergType, old: IcebergType | None) -> IcebergType:
        if isinstance(new, StructType):
            return struct(new, old if isinstance(old, StructType) else None)
        if isinstance(new, ListType):
            old_list = old if isinstance(old, ListType) else None
            keep = old_list is not None and _evolves(
                old_list.element_type, new.element_type
            )
            return ListType(
                element_id=old_list.element_id if keep and old_list else new_id(),
                element=typed(
                    new.element_type,
                    old_list.element_type if keep and old_list else None,
                ),
                element_required=new.element_required,
            )
        if isinstance(new, MapType):
            old_map = old if isinstance(old, MapType) else None
            # a map key cannot change type
            keep_key = old_map is not None and _same_type(
                old_map.key_type, new.key_type
            )
            keep_value = old_map is not None and _evolves(
                old_map.value_type, new.value_type
            )
            key_id = old_map.key_id if keep_key and old_map else new_id()
            key_type = typed(
                new.key_type, old_map.key_type if keep_key and old_map else None
            )
            value_id = old_map.value_id if keep_value and old_map else new_id()
            value_type = typed(
                new.value_type, old_map.value_type if keep_value and old_map else None
            )
            return MapType(
                key_id=key_id,
                key_type=key_type,
                value_id=value_id,
                value_type=value_type,
                value_required=new.value_required,
            )
        return new

    fields = struct(schema.as_struct(), metadata.schema().as_struct()).fields
    candidate = Schema(*fields)
    # the current schema first: a replace that keeps the columns adds no schema
    schemas = sorted(
        metadata.schemas, key=lambda s: s.schema_id != metadata.current_schema_id
    )
    for existing in schemas:
        if existing == candidate:
            return existing
    schema_id = max(s.schema_id for s in metadata.schemas) + 1
    return Schema(*fields, schema_id=schema_id)


def _same_type(a: IcebergType, b: IcebergType) -> bool:
    """Whether `a` and `b` are the same type, whatever their field ids."""
    from pyiceberg.types import ListType, MapType, StructType

    if isinstance(a, StructType) and isinstance(b, StructType):
        return len(a.fields) == len(b.fields) and all(
            x.name == y.name
            and x.required == y.required
            and _same_type(x.field_type, y.field_type)
            for x, y in zip(a.fields, b.fields, strict=True)
        )
    if isinstance(a, ListType) and isinstance(b, ListType):
        return a.element_required == b.element_required and _same_type(
            a.element_type, b.element_type
        )
    if isinstance(a, MapType) and isinstance(b, MapType):
        return (
            a.value_required == b.value_required
            and _same_type(a.key_type, b.key_type)
            and _same_type(a.value_type, b.value_type)
        )
    return a == b


def _evolves(old: IcebergType, new: IcebergType) -> bool:
    """Whether a field of type `old` can become one of type `new`, keeping its id."""
    from pyiceberg.types import (
        DecimalType,
        DoubleType,
        FloatType,
        IntegerType,
        ListType,
        LongType,
        MapType,
        StructType,
    )

    # Iceberg v3 restricts promotions of partition sources; v3 is not replaced
    if isinstance(new, (StructType, ListType, MapType)):
        return type(old) is type(new)
    if old == new:
        return True
    if isinstance(old, IntegerType) and isinstance(new, LongType):
        return True
    if isinstance(old, FloatType) and isinstance(new, DoubleType):
        return True
    return (
        isinstance(old, DecimalType)
        and isinstance(new, DecimalType)
        and old.scale == new.scale
        and new.precision >= old.precision
    )


def _replacement_spec(
    staged: TableMetadata, partition_by: list[tuple[list[str], str]], name: str
) -> pyiceberg.partitioning.PartitionSpec | int | None:
    """The partition spec change of a replace, on `staged` (with the new schema)."""
    from pyiceberg.transforms import parse_transform

    from polars.io.iceberg._partition_evolution import (
        _identity,
        _is_active,
        partition_source,
        spec_change,
    )

    if staged.format_version >= 2:
        return spec_change(staged, partition_by, name)
    # v1 specs keep removed fields (void), which the sink cannot write: only an
    # unchanged partitioning without them is replaced
    schema = staged.schema()
    target = []
    for path, transform_name in partition_by:
        transform = parse_transform(transform_name)
        target.append(
            (partition_source(schema, path, transform, name).field_id, transform)
        )
    fields = staged.spec().fields
    if not all(_is_active(f) for f in fields) or target != [
        _identity(f) for f in fields
    ]:
        msg = (
            f"cannot change the partitioning of table '{name}' in a replace: it is an "
            "Iceberg v1 table; upgrade it to v2 first, e.g. with PyIceberg: `with "
            "table.transaction() as tx: tx.upgrade_table_version(2)`, or "
            f"{_DROP_CREATE}"
        )
        raise InvalidOperationError(msg)
    return None


def _replacement_mapping(mappings: list[str | None], schema: Schema, name: str) -> str:
    """
    The name mappings `mappings` (JSON) merged, with the columns of `schema` added.

    Raises if a name maps to two fields: the mapping reads older data files without
    field ids by name.
    """
    from pyiceberg.table.name_mapping import (
        NameMapping,
        create_mapping_from_schema,
        parse_mapping_from_json,
    )

    merged: list[MappedField] = []
    for mapping in mappings:
        if mapping is None:
            continue
        try:
            parsed = list(parse_mapping_from_json(mapping))
        # pydantic's ValidationError
        except ValueError as error:
            msg = f"invalid name mapping for table '{name}': {error}"
            raise InvalidOperationError(msg) from error
        merged = _merge_mapped(merged, parsed, name)
    merged = _merge_mapped(merged, list(create_mapping_from_schema(schema)), name)
    # readers look a field up by its id, anywhere in the mapping
    seen: set[int | None] = set()
    stack = list(merged)
    while stack:
        m = stack.pop()
        if m.field_id is not None and m.field_id in seen:
            msg = (
                f"the name mapping of table '{name}' maps field id {m.field_id} in two "
                f"places; {_DROP_CREATE}"
            )
            raise DuplicateError(msg)
        seen.add(m.field_id)
        stack.extend(m.fields)
    return NameMapping(merged).model_dump_json()


def _label(names: list[str], field_id: int | None) -> str:
    """A mapped field's name in messages; Iceberg allows a field without names."""
    return names[0] if names else f"<field {field_id}>"


def _merge_mapped(
    old: list[MappedField], new: list[MappedField], name: str, path: str = ""
) -> list[MappedField]:
    """`old` with the entries of `new`: the names of a field id are joined."""

    def same(x: MappedField, m: MappedField) -> bool:
        # an entry without a field id is the same one only under the same names
        if m.field_id is None:
            return x.field_id is None and set(x.names) == set(m.names)
        return x.field_id == m.field_id

    out = list(old)
    for m in new:
        owners = {alias: x for x in out for alias in x.names}
        for alias in m.names:
            if alias in owners and not same(owners[alias], m):
                msg = (
                    f"column name '{path}{alias}' of table '{name}' is mapped to two "
                    "columns by its name mapping, which reads older data files with "
                    f"it; {_DROP_CREATE}"
                )
                raise DuplicateError(msg)
        i = next((i for i, x in enumerate(out) if same(x, m)), None)
        if i is None:
            # its children checked as well
            children = _merge_mapped(
                [], m.fields, name, f"{path}{_label(m.names, m.field_id)}."
            )
            out.append(m.model_copy(update={"fields": children}))
            continue
        names = out[i].names + [alias for alias in m.names if alias not in out[i].names]
        fields = _merge_mapped(
            out[i].fields, m.fields, name, f"{path}{_label(names, out[i].field_id)}."
        )
        out[i] = out[i].model_copy(update={"names": names, "fields": fields})
    return out
