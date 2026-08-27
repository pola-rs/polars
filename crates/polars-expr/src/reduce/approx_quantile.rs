use std::fmt;

use polars_compute::approx_quantile::req::ReqSketch;
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
    use VecGroupedReduction as VGR;
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
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(dtype, R::<Int128Type>::new(error))),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::new_idxsize(1))),
        _ => {
            polars_bail!(InvalidOperation: "`approx_n_unique` operation not supported for dtype `{dtype}`")
        },
    })
}

struct ApproxQuantileReducer<T> {
    error: f64,
    marker: PhantomData<T>,
}

impl<T> ApproxQuantileReducer<T> {
    fn new(error: f64) -> Self {
        Self {
            error,
            marker: PhantomData,
        }
    }
}

impl<T> Clone for ApproxQuantileReducer<T> {
    fn clone(&self) -> Self {
        Self {
            error: self.error,
            marker: PhantomData,
        }
    }
}

impl<T> Reducer for ApproxQuantileReducer<T>
where
    T: PolarsNumericType,
    T::Native: Clone + TotalOrd + fmt::Debug,
{
    type Dtype = T;
    type Value = ReqSketch<T::Native>;

    fn init(&self) -> Self::Value {
        ReqSketch::new(self.error, true)
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        todo!()
    }

    fn reduce_one(
        &self,
        sketch: &mut Self::Value,
        value: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        _seq_id: u64,
    ) {
        if let Some(value) = value {
            sketch.update(&[value]);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        todo!()
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        todo!()
    }
}
