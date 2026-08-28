from __future__ import annotations

from typing import TYPE_CHECKING

from polars._utils.wrap import wrap_expr

if TYPE_CHECKING:
    from polars import Expr


class ExprUuidNameSpace:
    """Namespace for UUID-related expressions."""

    _accessor = "uuid"

    def __init__(self, expr: Expr) -> None:
        self._pyexpr = expr._pyexpr

    def version(self) -> Expr:
        """
        Extract the four-bit UUID version field.

        Returns
        -------
        Expr
            Expression with dtype :class:`UInt8`. Null inputs remain null.

        Examples
        --------
        >>> from uuid import UUID
        >>> df = pl.DataFrame(
        ...     {"id": [UUID("67e55044-10b1-426f-9247-bb680e5fe0c8"), None]}
        ... )
        >>> df.select(pl.col("id").uuid.version()).to_series().to_list()
        [4, None]

        Notes
        -----
        Returns null where the input UUID is null.
        """
        return wrap_expr(self._pyexpr.uuid_version())

    def timestamp(self, *, strict: bool = True) -> Expr:
        """
        Extract the UTC millisecond timestamp encoded by UUIDv7.

        Parameters
        ----------
        strict
            Raise on a non-v7 UUID. If false, return null for non-v7 values.

        Returns
        -------
        Expr
            Expression with dtype ``Datetime("ms", time_zone="UTC")``.

        Examples
        --------
        >>> from uuid import UUID
        >>> df = pl.DataFrame({"id": [UUID("01890d97-ee80-7000-8000-000000000000")]})
        >>> result = df.select(pl.col("id").uuid.timestamp()).to_series()
        >>> result.dtype
        Datetime(time_unit='ms', time_zone='UTC')
        """
        return wrap_expr(self._pyexpr.uuid_timestamp(strict))
