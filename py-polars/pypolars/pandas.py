# Some functions similar to the pandas API
from typing import Union, TextIO, Optional, List

from .frame import DataFrame


def get_dummies(df: DataFrame) -> DataFrame:
    return df.to_dummies()


def read_csv(
    file: Union[str, TextIO],
    infer_schema_length: int = 100,
    batch_size: int = 1000,
    has_headers: bool = True,
    ignore_errors: bool = False,
    stop_after_n_rows: Optional[int] = None,
    skip_rows: int = 0,
    projection: Optional[List[int]] = None,
    sep: str = ",",
    cols: Optional[List[str]] = None,
) -> "DataFrame":
    return DataFrame.read_csv(
        file=file,
        infer_schema_length=infer_schema_length,
        batch_size=batch_size,
        has_headers=has_headers,
        ignore_errors=ignore_errors,
        stop_after_n_rows=stop_after_n_rows,
        skip_rows=skip_rows,
        projection=projection,
        sep=sep,
        cols=cols,
    )
