import polars.internals as pli

try:
    from polars.polars import PySQLContext

    _DOCUMENTING = False
except ImportError:
    _DOCUMENTING = True


class SQLContext:
    """
    Run SQL query against a LazyFrame.

    Warnings
    --------
    This feature is experimental and may change without it being
    considered a breaking change.

    """

    def __init__(self) -> None:
        self._ctxt = PySQLContext.new()

    def register(self, name: str, lf: pli.LazyFrame) -> None:
        """
        Register a ``LazyFrame`` in this ``SQLContext`` under a given ``name``.

        Parameters
        ----------
        name
            Name of the table
        lf
            LazyFrame to add as this table name.

        """
        self._ctxt.register(name, lf._ldf)

    def execute(self, query: str) -> pli.LazyFrame:
        """
        Parse the givens SQL query and transform that to a ``LazyFrame``.

        Parameters
        ----------
        query
            A SQL query

        """
        return pli.wrap_ldf(self._ctxt.execute(query))

    def query(self, query: str) -> pli.DataFrame:
        return self.execute(query).collect()
