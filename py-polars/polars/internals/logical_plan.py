from io import BytesIO, IOBase, StringIO
from pathlib import Path
from typing import Optional, Union

from polars.utils import format_path

try:
    from polars.polars import PyLogicalPlan

    _DOCUMENTING = False
except ImportError:  # pragma: no cover
    _DOCUMENTING = True


class LogicalPlan:
    def __init__(self):
        self._lp = PyLogicalPlan

    @classmethod
    def _from_pylp(cls: "LogicalPlan", lp: "PyLogicalPlan") -> "LogicalPlan":
        self = cls.__new__(cls)
        self._lp = lp
        return self

    def write_json(
        self,
        file: Optional[Union[IOBase, str, Path]] = None,
        *,
        to_string: bool = False,
    ) -> Optional[str]:
        """
        Serialize to JSON representation.

        Parameters
        ----------
        file
            Write to this file instead of returning a string.
        to_string
            Ignore file argument and return a string.
        """
        if isinstance(file, (str, Path)):
            file = format_path(file)
        to_string_io = (file is not None) and isinstance(file, StringIO)
        if to_string or file is None or to_string_io:
            with BytesIO() as buf:
                self._lp.to_json(buf)
                json_bytes = buf.getvalue()

            json_str = json_bytes.decode("utf8")
            if to_string_io:
                file.write(json_str)  # type: ignore[union-attr]
            else:
                return json_str
        else:
            self._lp.to_json(file)
        return None


def wrap_lp(lp: PyLogicalPlan) -> LogicalPlan:
    return LogicalPlan._from_pylp(lp)
