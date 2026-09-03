use polars_core::prelude::*;
use polars_core::series::ops::NullBehavior;

use crate::prelude::diff;

pub fn pct_change(s: &Series, n: &Series) -> PolarsResult<Series> {
    polars_ensure!(
        n.len() == 1,
        ComputeError: "n must be a single value."
    );

    match s.dtype() {
        #[cfg(feature = "dtype-f16")]
        DataType::Float16 => {},
        DataType::Float64 | DataType::Float32 => {},
        dt => {
            let casted = s.cast(&DataType::Float64)?;
            polars_ensure!(casted.dtype() == &DataType::Float64, opq = pct_change, dt);
            return pct_change(&casted, n);
        },
    }

    let n_s = n.cast(&DataType::Int64)?;
    if let Some(n) = n_s.i64()?.get(0) {
        diff(s, n, NullBehavior::Ignore)?.divide(&s.shift(n))
    } else {
        Ok(Series::full_null(s.name().clone(), s.len(), s.dtype()))
    }
}
