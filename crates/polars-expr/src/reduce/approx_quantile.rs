use std::fmt;

use polars_compute::kll::KLLSketch;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;
use rand::SeedableRng;
use rand::rngs::SmallRng;

use super::*;

pub fn new_approx_quantile_reduction(
    dtype: DataType,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use ApproxQuantileReducer as R;
    use DataType::*;
    // TODO: [amber] blocker: Cannot use a VecGroupedReduction for this
    Ok(match dtype {
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, R::<$T>::new(error)))
            })
        },
        // TODO: [amber]
        // Boolean => Box::new(VGR::new(dtype, R::<BooleanType>::new(error))),
        // String => Box::new(VGR::new(dtype, R::<StringType>::new(error))),
        // Binary => Box::new(VGR::new(dtype, R::<BinaryType>::new(error))),
        // #[cfg(feature = "dtype-decimal")]
        // Decimal(_, _) => Box::new(VGR::new(dtype, R::<Int128Type>::new(error))),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::new_idxsize(1))),
        _ => {
            polars_bail!(InvalidOperation: "`approx_n_unique` operation not supported for dtype `{dtype}`")
        },
    })
}

struct ApproxQuantileGroupedReduction {}
