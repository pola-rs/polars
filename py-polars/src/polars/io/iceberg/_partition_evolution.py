"""The partition spec of `ALTER TABLE ... SET | RESET PARTITIONED BY` (Iceberg)."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Literal, NoReturn

from polars.exceptions import InvalidOperationError, SQLInterfaceError

if TYPE_CHECKING:
    import pyiceberg.partitioning
    import pyiceberg.schema
    import pyiceberg.table
    import pyiceberg.transforms
    from pyiceberg.table.metadata import TableMetadata

#: (source field id, transform): what a partition field partitions by.
_Identity = tuple[int, Any]
#: Iceberg field ids are signed 32-bit integers.
_MAX_FIELD_ID = 2**31 - 1


def partition_field_name(
    column: str, transform: pyiceberg.transforms.Transform[Any, Any]
) -> str:
    """The name of a new partition field, as PyIceberg's `update_spec` names it."""
    from pyiceberg.transforms import (
        BucketTransform,
        IdentityTransform,
        TruncateTransform,
    )

    if isinstance(transform, IdentityTransform):
        return column
    if isinstance(transform, BucketTransform):
        return f"{column}_bucket_{transform.num_buckets}"
    if isinstance(transform, TruncateTransform):
        return f"{column}_trunc_{transform.width}"
    return f"{column}_{transform}"


def partition_source(
    schema: pyiceberg.schema.Schema,
    path: list[str],
    transform: pyiceberg.transforms.Transform[Any, Any],
    name: str,
    *,
    kind: Literal["partition", "sort"] = "partition",
) -> Any:
    """The field at `path` (a column, then struct fields) that partitions or sorts."""
    from pyiceberg.transforms import TruncateTransform
    from pyiceberg.types import DecimalType, StructType

    column = ".".join(path)
    top = [f.name for f in schema.fields]
    if path[1:] and column in top:
        hint = f"; for the column named '{column}', quote it: \"{column}\""
    elif not path[1:] and "." in column and column.split(".")[0] in top:
        hint = f"; for a struct field, leave the dots unquoted: {column}"
    elif path[1:] and path[0] == name.split(".")[-1]:
        hint = "; table-qualified names are not supported"
    else:
        hint = ""
    # part by part: `find_field` would also take a column named `s.f` for `s`, `f`
    fields, source = schema.fields, None
    for i, part in enumerate(path):
        if (source := next((f for f in fields if f.name == part), None)) is None:
            if i == 0:
                msg = (
                    f"{kind} column '{column}' is not a column of table '{name}'"
                    f"{hint}; valid columns: {top}"
                )
            else:
                msg = (
                    f"{kind} column '{column}' is not a column of table '{name}': "
                    f"struct '{'.'.join(path[:i])}' has no field '{part}'"
                    f"{hint if column in top else ''}; its fields: "
                    f"{[f.name for f in fields]}"
                )
            raise SQLInterfaceError(msg)
        if path[i + 1 :]:
            # Iceberg nests partition and sort sources in structs, not lists or maps
            if not isinstance(source.field_type, StructType):
                msg = (
                    f"cannot {kind} table '{name}' by '{column}': "
                    f"'{'.'.join(path[: i + 1])}' has type {source.field_type}, and "
                    f"only struct fields can be {kind} sources"
                )
                raise SQLInterfaceError(msg)
            fields = source.field_type.fields
    assert source is not None  # the parser gives at least one part
    source_type = source.field_type
    if not source_type.is_primitive or not transform.can_transform(source_type):
        msg = (
            f"cannot {kind} table '{name}' by {transform} of column "
            f"'{column}' with type {source_type}"
        )
        raise SQLInterfaceError(msg)
    # the sink sorts by it, but does not write it as a partition value
    if (
        kind == "partition"
        and isinstance(transform, TruncateTransform)
        and isinstance(source_type, DecimalType)
    ):
        msg = f"partitioning by {transform} of a decimal column ('{column}')"
        raise NotImplementedError(msg)
    return source


def spec_change(
    metadata: TableMetadata, fields: list[tuple[list[str], str]], name: str
) -> pyiceberg.partitioning.PartitionSpec | int | None:
    """
    The change that makes `fields` (source path, transform) the default spec of `name`.

    A new spec, the id of an existing spec, or `None` if the default spec has them.
    """
    from pyiceberg.partitioning import PartitionField, PartitionSpec
    from pyiceberg.schema import index_by_name
    from pyiceberg.transforms import TimeTransform, parse_transform

    if metadata.format_version < 2:
        msg = (
            f"cannot change the partitioning of table '{name}': it is an Iceberg "
            f"v{metadata.format_version} table; upgrade it to v2 first, e.g. with "
            "PyIceberg: `with table.transaction() as tx: tx.upgrade_table_version(2)`"
        )
        raise InvalidOperationError(msg)
    if metadata.format_version > 2:
        msg = (
            "changing the partitioning of an Iceberg "
            f"v{metadata.format_version} table is not supported yet: table '{name}'"
        )
        raise NotImplementedError(msg)

    schema = metadata.schema()
    targets: list[tuple[_Identity, str]] = []
    for path, transform_name in fields:
        transform = parse_transform(transform_name)
        source = partition_source(schema, path, transform, name)
        targets.append(((source.field_id, transform), ".".join(path)))

    current = metadata.spec()
    kept = {_identity(f): f for f in current.fields if _is_active(f)}
    # newest spec first: a field id's newest name wins
    specs = sorted(
        metadata.partition_specs, key=lambda spec: spec.spec_id, reverse=True
    )
    history: dict[_Identity, list[PartitionField]] = {}
    # what each field id partitions by; v1 tables give an id per position, per spec
    id_identities: dict[int, set[_Identity]] = {}
    for spec in specs:
        for field in spec.fields:
            if _is_active(field):
                history.setdefault(_identity(field), []).append(field)
                id_identities.setdefault(field.field_id, set()).add(_identity(field))

    # Iceberg's spec builders take one new time transform per column
    new_time: dict[int, int] = {}
    for i, (identity, _) in enumerate(targets):
        if identity not in kept and isinstance(identity[1], TimeTransform):
            if (earlier := new_time.setdefault(identity[0], i)) != i:
                column = targets[i][1]
                msg = (
                    f"partition fields `{fields[earlier][1]}({column})` and "
                    f"`{fields[i][1]}({column})` both newly partition column "
                    f"'{column}' of table '{name}' by time; add one of them at a time"
                )
                raise SQLInterfaceError(msg)

    # Field ids: kept fields keep theirs, reintroduced ones reuse their highest id that
    # no other field had, new ones count on from the highest id in any spec.
    last_id = max(
        [metadata.last_partition_id or 999]
        + [f.field_id for spec in specs for f in spec.fields]
    )
    next_id = last_id
    ids: list[int] = []
    for identity, _ in targets:
        if identity in kept:
            ids.append(kept[identity].field_id)
        elif own := [
            f.field_id
            for f in history.get(identity, [])
            if id_identities[f.field_id] == {identity}
        ]:
            ids.append(max(own))
        else:
            next_id += 1
            ids.append(next_id)

    # Names: kept fields keep theirs. Then reintroduced ones (newest name) and new
    # ones take a name that no column has and no field of another column had.
    columns = index_by_name(schema)
    used_by: dict[str, set[_Identity]] = {}
    for spec in specs:
        for field in spec.fields:
            used = used_by.setdefault(field.name, set())
            if _is_active(field):
                used.add(_identity(field))
            else:
                # a removed v1 field: what its id partitioned by
                used |= id_identities.get(field.field_id, set())
    names: dict[int, str] = {
        i: kept[identity].name
        for i, (identity, _) in enumerate(targets)
        if identity in kept
    }

    def free(candidate: str, identity: _Identity) -> bool:
        return (
            candidate not in names.values()
            and not _clashes(candidate, identity, columns)
            and used_by.get(candidate, set()) <= {identity}
        )

    for i, (identity, column) in enumerate(targets):
        if i in names:
            continue
        base = partition_field_name(column, identity[1])
        candidates = [base, f"{base}_{ids[i]}"]
        newest = [f for f in history.get(identity, []) if f.field_id == ids[i]]
        if newest:
            candidates.insert(0, newest[0].name)
        n = 2
        while not any(free(c, identity) for c in candidates):
            candidates.append(f"{base}_{ids[i]}_{n}")
            n += 1
        names[i] = next(c for c in candidates if free(c, identity))

    target = [
        PartitionField(
            source_id=identity[0],
            field_id=ids[i],
            transform=identity[1],
            name=names[i],
        )
        for i, (identity, _) in enumerate(targets)
    ]

    # An existing spec with these fields is reused, as Iceberg does. One that keeps the
    # kept fields' ids first, else the newest.
    compatible = [spec for spec in specs if _compatible(spec, target)]
    if any(spec.spec_id == current.spec_id for spec in compatible):
        return None
    keeping = [
        spec
        for spec in compatible
        if all(
            f.field_id == t.field_id
            for f, t in zip(spec.fields, target, strict=True)
            if _identity(t) in kept
        )
    ]
    if compatible:
        return (keeping or compatible)[0].spec_id
    _validate(target, metadata, next_id, {f.field_id for f in kept.values()}, name)
    spec_id = max(spec.spec_id for spec in specs) + 1
    return PartitionSpec(*target, spec_id=spec_id)


def _identity(field: pyiceberg.partitioning.PartitionField) -> _Identity:
    return (field.source_id, field.transform)


def _is_active(field: pyiceberg.partitioning.PartitionField) -> bool:
    from pyiceberg.transforms import VoidTransform

    return not isinstance(field.transform, VoidTransform)


def _clashes(name: str, identity: _Identity, columns: dict[str, int]) -> bool:
    """Whether field `name` takes a column's name, other than its own column's."""
    from pyiceberg.transforms import IdentityTransform

    source_id, transform = identity
    return name in columns and not (
        isinstance(transform, IdentityTransform) and columns[name] == source_id
    )


def _compatible(
    spec: pyiceberg.partitioning.PartitionSpec,
    fields: list[pyiceberg.partitioning.PartitionField],
) -> bool:
    """Whether `spec` has `fields` in order, by source, transform and name."""
    return len(spec.fields) == len(fields) and all(
        (f.source_id, f.transform, f.name) == (g.source_id, g.transform, g.name)
        for f, g in zip(spec.fields, fields, strict=True)
    )


def _validate(
    fields: list[pyiceberg.partitioning.PartitionField],
    metadata: TableMetadata,
    last_id: int,
    kept_ids: set[int],
    name: str,
) -> None:
    """
    Check the fields of a new spec, which raw metadata updates do not.

    Fields with an id in `kept_ids` come from the current spec, as they are.
    """
    from pyiceberg.schema import index_by_name
    from pyiceberg.transforms import TruncateTransform
    from pyiceberg.types import DecimalType

    schema = metadata.schema()
    columns = index_by_name(schema)

    def refuse(field: pyiceberg.partitioning.PartitionField, why: str) -> NoReturn:
        msg = (
            f"cannot change the partitioning of table '{name}': partition field "
            f"'{field.name}' {why}"
        )
        raise InvalidOperationError(msg)

    for i, field in enumerate(fields):
        try:
            source_type = schema.find_type(field.source_id)
        except ValueError:
            refuse(field, "has no source column")
        if not field.transform.can_transform(source_type):
            refuse(field, f"cannot partition by {field.transform} of {source_type}")
        if isinstance(field.transform, TruncateTransform) and isinstance(
            source_type, DecimalType
        ):
            refuse(field, "truncates a decimal column, which is not supported")
        if not _is_active(field):
            refuse(field, "is void")
        if not field.name:
            refuse(field, "has no name")
        if _clashes(field.name, _identity(field), columns):
            refuse(field, "has the name of a column")
        kept = field.field_id in kept_ids
        top = min(last_id, _MAX_FIELD_ID)
        if not kept and not 1000 <= field.field_id <= top:
            refuse(field, f"has field id {field.field_id}, outside 1000-{top}")
        for other in fields[:i]:
            if other.name == field.name or other.field_id == field.field_id:
                refuse(field, "repeats the name or field id of another field")
            if _identity(other) == _identity(field):
                refuse(field, "repeats another field")
        for spec in [] if kept else metadata.partition_specs:
            for other in spec.fields:
                if (
                    other.field_id == field.field_id
                    and _is_active(other)
                    and _identity(other) != _identity(field)
                ):
                    refuse(
                        field,
                        f"has field id {field.field_id}, which partitions by "
                        f"{other.transform} of another column in partition spec "
                        f"{spec.spec_id}",
                    )


def commit_spec_change(
    table: pyiceberg.table.Table,
    metadata: TableMetadata,
    change: pyiceberg.partitioning.PartitionSpec | int,
) -> None:
    """Commit `change` against `metadata`, failing if the table changed since."""
    from pyiceberg.partitioning import PartitionSpec
    from pyiceberg.table.update import (
        AddPartitionSpecUpdate,
        AssertCurrentSchemaId,
        AssertDefaultSpecId,
        AssertLastAssignedPartitionId,
        SetDefaultSpecUpdate,
    )

    updates: tuple[Any, ...]
    if isinstance(change, PartitionSpec):
        # -1: the spec just added, whichever id the catalog gives it
        updates = (
            AddPartitionSpecUpdate(spec=change),
            SetDefaultSpecUpdate(spec_id=-1),  # type: ignore[call-arg]
        )
    else:
        updates = (SetDefaultSpecUpdate(spec_id=change),)  # type: ignore[call-arg]
    requirements = (
        AssertDefaultSpecId(default_spec_id=metadata.default_spec_id),  # type: ignore[call-arg]
        AssertLastAssignedPartitionId(  # type: ignore[call-arg]
            last_assigned_partition_id=metadata.last_partition_id
        ),
        AssertCurrentSchemaId(current_schema_id=metadata.current_schema_id),  # type: ignore[call-arg]
    )
    tx = table.transaction()
    tx._stage(updates, requirements)
    tx.commit_transaction()
