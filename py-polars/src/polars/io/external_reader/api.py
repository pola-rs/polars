from polars._typing import FileSource, SchemaDict
from polars._utils.wrap import wrap_ldf
from polars.io.external_reader._external_reader import FileReaderBuilder
from polars.io.scan_options._options import ScanOptions
from polars.lazyframe.frame import LazyFrame


def scan_external_reader(
    reader_builder: FileReaderBuilder,
    *,
    sources: FileSource,
    schema: SchemaDict,
    expand_paths: bool = False,
) -> LazyFrame:
    """
    Scan an external reader.

    Parameters
    ----------
    reader_builder
        The external reader builder.
    sources
        The sources to scan.
    schema
        The schema of the data.
    expand_paths
        Whether to expand provided paths.

    Returns
    -------
    LazyFrame
        A lazy frame representing the scanned data.
    """
    from polars._plr import PyLazyFrame

    return wrap_ldf(
        PyLazyFrame.new_from_external_reader_builder(
            sources=sources,
            reader_builder=reader_builder,
            schema=schema,
            scan_options=ScanOptions(expand_paths=expand_paths),
        )
    )
