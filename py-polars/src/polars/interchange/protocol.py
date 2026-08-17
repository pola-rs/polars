from __future__ import annotations

from polars._utils.unstable import issue_unstable_warning


class CompatLevel:
    """Data structure compatibility level."""

    _version: int

    def __init__(self) -> None:
        msg = "it is not allowed to create a CompatLevel object"
        raise TypeError(msg)

    @staticmethod
    def _with_version(version: int) -> CompatLevel:
        compat_level = CompatLevel.__new__(CompatLevel)
        compat_level._version = version
        return compat_level

    @staticmethod
    def _newest() -> CompatLevel:
        return CompatLevel._future1  # type: ignore[attr-defined]

    @staticmethod
    def newest() -> CompatLevel:
        """
        Get the highest supported compatibility level.

        .. warning::
            Highest compatibility level is considered **unstable**. It may be changed
            at any point without it being considered a breaking change.
        """
        issue_unstable_warning(
            "using the highest compatibility level is considered unstable."
        )
        return CompatLevel._newest()

    @staticmethod
    def oldest() -> CompatLevel:
        """Get the most compatible level."""
        return CompatLevel._compatible  # type: ignore[attr-defined]

    def __repr__(self) -> str:
        return f"<{self.__class__.__module__}.{self.__class__.__qualname__}: {self._version}>"


CompatLevel._compatible = CompatLevel._with_version(0)  # type: ignore[attr-defined]
CompatLevel._future1 = CompatLevel._with_version(1)  # type: ignore[attr-defined]
