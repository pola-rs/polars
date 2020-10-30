# Some functions similar to the pandas API
from typing import Union, TextIO, Optional

from .frame import DataFrame


def get_dummies(df: DataFrame) -> DataFrame:
    return df.to_dummies()


def read_csv(
    file: Union[str, TextIO],
    infer_schema_length: int = 100,
    batch_size: int = 100000,
    has_headers: bool = True,
    ignore_errors: bool = False,
    stop_after_n_rows: Optional[int] = None,
    sep: str = ",",
) -> "DataFrame":
    return DataFrame.read_csv(
        file,
        infer_schema_length,
        batch_size,
        has_headers,
        ignore_errors,
        stop_after_n_rows,
        sep,
    )
