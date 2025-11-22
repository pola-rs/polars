from .io_plugin import RandomSource, new_bernoulli, new_uniform
from typing import Any, Iterator
from polars.io.plugins import register_io_source
import polars as pl

__all__ = ["RandomSource", "new_bernoulli", "new_uniform", "scan_random"]


def scan_random(samplers: list[Any], size: int = 1000) -> pl.LazyFrame:
    def source_generator(
        with_columns: list[str] | None,
        predicate: pl.Expr | None,
        n_rows: int | None,
        batch_size: int | None,
    ) -> Iterator[pl.DataFrame]:
        """
        Generator function that creates the source.
        This function will be registered as IO source.
        """

        new_size = size
        if n_rows is not None and n_rows < size:
            new_size = n_rows

        src = RandomSource(samplers, batch_size, new_size)
        if with_columns is not None:
            src.set_with_columns(with_columns)

        # Set the predicate.
        predicate_set = True
        if predicate is not None:
            try:
                src.try_set_predicate(predicate)
            except pl.exceptions.ComputeError:
                predicate_set = False

        while (out := src.next()) is not None:
            # If the source could not apply the predicate
            # (because it wasn't able to deserialize it), we do it here.
            if not predicate_set and predicate is not None:
                out = out.filter(predicate)

            yield out

    # create src again to compute the schema
    src = RandomSource(samplers, 0, 0)
    return register_io_source(io_source=source_generator, schema=src.schema())
