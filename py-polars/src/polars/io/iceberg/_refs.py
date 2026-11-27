"""Tags and branches of Iceberg tables, set and dropped by SQL `ALTER TABLE`."""

from __future__ import annotations

from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    import pyiceberg.table


def change_ref(
    table: pyiceberg.table.Table, op: dict[str, Any], name: str, *, note: str
) -> None:
    """
    Set or drop the tag or branch of `op` on `table`, in one commit.

    Nothing is committed when the ref would not change. `name` is the table as the
    statement wrote it, `note` is added to an error of the commit.
    """
    from pyiceberg.table.refs import MAIN_BRANCH

    from polars.exceptions import DuplicateError, InvalidOperationError
    from polars.io.iceberg._time_travel import resolve_version

    kind, ref_name = op["ref_kind"], op["ref"]
    # as loaded: the commit asserts that the ref still has this snapshot
    existing = table.metadata.refs.get(ref_name)
    existing_kind = (
        None if existing is None else existing.snapshot_ref_type.value.lower()
    )

    if op["op"] == "drop_ref":
        if existing is None:
            if op["if_exists"]:
                return
            msg = f"{kind} '{ref_name}' of table '{name}' does not exist"
            raise InvalidOperationError(msg)
        if existing_kind != kind:
            msg = f"'{ref_name}' of table '{name}' is a {existing_kind}, not a {kind}"
            raise InvalidOperationError(msg)
        if ref_name == MAIN_BRANCH:
            msg = f"the main branch of table '{name}' cannot be dropped"
            raise InvalidOperationError(msg)
        manage = table.manage_snapshots()
        if kind == "tag":
            manage.remove_tag(ref_name)
        else:
            manage.remove_branch(ref_name)
        _commit(manage, note)
        return

    mode = op["mode"]
    # a taken name, of either kind, makes `IF NOT EXISTS` a no-op
    if existing is not None and mode == "create_if_not_exists":
        return
    if ref_name == MAIN_BRANCH and (
        kind == "tag" or mode in ("replace", "create_or_replace")
    ):
        msg = (
            f"'{MAIN_BRANCH}' is the main branch of table '{name}': it cannot be "
            "a tag or be replaced"
        )
        raise InvalidOperationError(msg)
    if existing is not None:
        if mode == "create":
            msg = (
                f"a {existing_kind} named '{ref_name}' already exists on table '{name}'"
            )
            raise DuplicateError(msg)
        if existing_kind != kind:
            msg = f"'{ref_name}' of table '{name}' is a {existing_kind}, not a {kind}"
            raise InvalidOperationError(msg)
    elif mode == "replace":
        msg = f"{kind} '{ref_name}' of table '{name}' does not exist"
        raise InvalidOperationError(msg)

    if op["snapshot_id"] is None:
        current = table.current_snapshot()
        if current is None:
            msg = f"table '{name}' has no snapshot for a {kind} to point at"
            raise InvalidOperationError(msg)
        snapshot_id = current.snapshot_id
    else:
        try:
            snapshot_id = resolve_version(table, ("snapshot", op["snapshot_id"]))
        except ValueError as error:
            error.add_note(f"`AS OF VERSION` of {kind} '{ref_name}' on table '{name}'")
            raise

    # a replaced ref keeps the retention the statement does not set
    def retention(key: str) -> int | None:
        if op.get(key) is not None:
            return op[key]
        return None if existing is None else getattr(existing, key)

    max_ref_age_ms = retention("max_ref_age_ms")
    if kind == "tag":
        branch_retention: tuple[int | None, int | None] = (None, None)
    else:
        branch_retention = (
            retention("max_snapshot_age_ms"),
            retention("min_snapshots_to_keep"),
        )
    if (
        existing is not None
        and existing.snapshot_id == snapshot_id
        and existing.max_ref_age_ms == max_ref_age_ms
        and (existing.max_snapshot_age_ms, existing.min_snapshots_to_keep)
        == branch_retention
    ):
        return

    manage = table.manage_snapshots()
    if kind == "tag":
        manage.create_tag(snapshot_id, ref_name, max_ref_age_ms=max_ref_age_ms)
    else:
        max_snapshot_age_ms, min_snapshots_to_keep = branch_retention
        manage.create_branch(
            snapshot_id,
            ref_name,
            max_ref_age_ms=max_ref_age_ms,
            max_snapshot_age_ms=max_snapshot_age_ms,
            min_snapshots_to_keep=min_snapshots_to_keep,
        )
    _commit(manage, note)


def _commit(manage: Any, note: str) -> None:
    try:
        manage.commit()
    except Exception as error:
        error.add_note(note)
        raise
