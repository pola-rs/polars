use std::fmt;
use std::marker::PhantomData;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::series::sketches_to_series;
use polars_utils::total_ord::TotalOrd;

use super::*;

pub fn new_approx_quantile_sketch_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    use SketchReducer as R;
    use VecGroupedReduction as VGR;
    Ok(match dtype {
        DataType::Boolean => Box::new(VGR::new(dtype, R::<BooleanType, bool>::new(method, error))),
        DataType::String => Box::new(VGR::new(dtype, R::<StringType, String>::new(method, error))),
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() || dtype.is_decimal() => {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                type Item<$T> = <$T as PolarsNumericType>::Native;
                Box::new(VGR::new(dtype, R::<$T, Item<$T>>::new(method, error)))
            })
        },
        _ => {
            polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{dtype}`")
        },
    })
}

struct SketchReducer<T, I: fmt::Debug + Clone + TotalOrd> {
    template: Sketch<I>,
    dtype: PhantomData<T>,
}

impl<T, I: fmt::Debug + Clone + TotalOrd> SketchReducer<T, I> {
    fn new(method: ApproxQuantileMethod, error: f64) -> Self {
        Self {
            template: Sketch::new(&method, error),
            dtype: PhantomData,
        }
    }
}

impl<T, I: fmt::Debug + Clone + TotalOrd> Clone for SketchReducer<T, I> {
    fn clone(&self) -> Self {
        Self {
            template: self.template.clone(),
            dtype: PhantomData,
        }
    }
}

impl<T, I> Reducer for SketchReducer<T, I>
where
    T: PolarsPhysicalType,
    I: for<'a> From<T::Physical<'a>>
        + fmt::Debug
        + Clone
        + TotalOrd
        + Send
        + Sync
        + serde::Serialize
        + 'static,
{
    type Dtype = T;
    type Value = Sketch<I>;

    fn init(&self) -> Self::Value {
        self.template.clone()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.merge(b);
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Physical<'_>>, _seq_id: u64) {
        if let Some(b) = b {
            a.update_owned(I::from(b));
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<T>, _seq_id: u64) {
        for value in ca.iter().flatten() {
            v.update_owned(I::from(value));
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let sketches: Vec<_> = v.into_iter().map(Sketch::finalize).collect();
        sketches_to_series(&sketches)
    }
}
