use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::chunked_array::peaks::{peak_max as pmax, peak_min as pmin};

use super::*;

pub(super) fn peak_min(s: &Series) -> PolarsResult<Series> {
    let s = s.to_physical_repr();
    let s = match s.dtype() {
        DataType::Boolean => polars_bail!(opq = peak_min, DataType::Boolean),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => pmin(s.decimal()?).into_series(),
        dt => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                pmin(ca).into_series()
            })
        },
    };
    Ok(s)
}

pub(super) fn peak_max(s: &Series) -> PolarsResult<Series> {
    let s = s.to_physical_repr();
    let s = match s.dtype() {
        DataType::Boolean => polars_bail!(opq = peak_max, DataType::Boolean),
        #[cfg(feature = "dtype-decimal")]
        DataType::Decimal(_, _) => pmax(s.decimal()?).into_series(),
        dt => {
            with_match_physical_numeric_polars_type!(dt, |$T| {
                let ca: &ChunkedArray<$T> = s.as_ref().as_ref().as_ref();
                pmax(ca).into_series()
            })
        },
    };
    Ok(s)
}
