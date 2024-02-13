use std::ops::Not;

use polars_core::with_match_physical_integer_polars_type;

use super::*;

pub fn negate_bitwise(s: &Series) -> PolarsResult<Series> {
    match s.dtype() {
        DataType::Boolean => Ok(s.bool().unwrap().not().into_series()),
        dt if dt.is_integer() => {
            with_match_physical_integer_polars_type!(dt, |$T| {
                    let ca: &ChunkedArray<$T> = s.as_any().downcast_ref().unwrap();
                    Ok(ca.apply_values(|v| !v).into_series())
            })
        },
        dt => polars_bail!(InvalidOperation: "dtype {:?} not supported in 'not' operation", dt),
    }
}
