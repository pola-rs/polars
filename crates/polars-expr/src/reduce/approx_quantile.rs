use std::fmt;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::total_ord::TotalOrd;

use super::*;

pub fn new_approx_quantile_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    // TODO: Move the error checks up and make this function infallible
    use ApproxQuantileReducer as R;
    use DataType::*;
    use VecGroupedReduction as VGR;
    Ok(match dtype {
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, R::<$T>::new(method.clone(), error)))
            })
        },
        // TODO: [amber]
        // Boolean => Box::new(VGR::new(dtype, R::<BooleanType>::new(error))),
        // String => Box::new(VGR::new(dtype, R::<StringType>::new(error))),
        // Binary => Box::new(VGR::new(dtype, R::<BinaryType>::new(error))),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VGR::new(dtype, R::<Int128Type>::new(method.clone(), error))),
        Null => Box::new(super::NullGroupedReduction::new(Scalar::new_idxsize(1))),
        _ => {
            polars_bail!(InvalidOperation: "`approx_n_unique` operation not supported for dtype `{dtype}`")
        },
    })
}

struct ApproxQuantileReducer<T> {
    method: ApproxQuantileMethod,
    error: f64,
    marker: PhantomData<T>,
}

impl<T> ApproxQuantileReducer<T> {
    fn new(method: ApproxQuantileMethod, error: f64) -> Self {
        Self {
            method,
            error,
            marker: PhantomData,
        }
    }
}

impl<T> Clone for ApproxQuantileReducer<T> {
    fn clone(&self) -> Self {
        Self {
            method: self.method.clone(),
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
    type Value = Sketch<T::Native>;

    fn init(&self) -> Self::Value {
        Sketch::new(&self.method, self.error)
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
