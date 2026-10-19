use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::series::sketches_to_series;
use polars_utils::total_ord::TotalOrd;

use super::split::{SplitStage, merge_reduction, split_reduction};
use super::*;

/// Runs `$body` with `$T` the physical type reduced for `$dtype` and `$B` the
/// borrowed type of the sketch items.
macro_rules! with_match_sketch_item {
    ($dtype:expr, | $T:ident, $B:ident | $body:expr) => {{
        let dtype: &DataType = $dtype;
        match dtype {
            DataType::Boolean => {
                type $T = BooleanType;
                type $B = bool;
                $body
            },
            DataType::String => {
                type $T = StringType;
                type $B = str;
                $body
            },
            _ if dtype.is_primitive_numeric() || dtype.is_temporal() || dtype.is_decimal() => {
                with_match_physical_numeric_polars_type!(dtype.to_physical(), |$P| {
                    type $T = $P;
                    type $B = <$P as PolarsNumericType>::Native;
                    $body
                })
            },
            _ => {
                polars_bail!(InvalidOperation: "`approx_quantile` operation not supported for dtype `{dtype}`")
            },
        }
    }};
}

pub fn new_approx_quantile_sketch_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    sketch_reduction(dtype, method, error, SplitStage::Whole)
}

/// Like [`new_approx_quantile_sketch_reduction`], but outputs the serialized
/// ingesting state of each group, to be merged by
/// [`new_approx_quantile_merge_reduction`].
pub fn new_approx_quantile_state_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    sketch_reduction(dtype, method, error, SplitStage::State)
}

/// Merges the states of [`new_approx_quantile_state_reduction`] over values of
/// `values_dtype`, and outputs the same sketches as
/// [`new_approx_quantile_sketch_reduction`].
pub fn new_approx_quantile_merge_reduction(
    values_dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    sketch_reduction(values_dtype, method, error, SplitStage::Merge)
}

fn sketch_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
    stage: SplitStage,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&dtype, |T, B| {
        let reducer = SketchReducer::<T, B>::new(method, error);
        match stage {
            SplitStage::Merge => merge_reduction(dtype.clone(), reducer),
            _ => split_reduction(dtype.clone(), reducer, stage),
        }
    }))
}

struct SketchReducer<T, B: ToOwned + ?Sized>
where
    B::Owned: fmt::Debug + Clone + TotalOrd,
{
    template: Sketch<B::Owned>,
    dtype: PhantomData<T>,
}

impl<T, B: ToOwned + ?Sized> SketchReducer<T, B>
where
    B::Owned: fmt::Debug + Clone + TotalOrd,
{
    fn new(method: ApproxQuantileMethod, error: f64) -> Self {
        Self {
            template: Sketch::new(&method, error),
            dtype: PhantomData,
        }
    }
}

impl<T, B: ToOwned + ?Sized> Clone for SketchReducer<T, B>
where
    B::Owned: fmt::Debug + Clone + TotalOrd,
{
    fn clone(&self) -> Self {
        Self {
            template: self.template.clone(),
            dtype: PhantomData,
        }
    }
}

impl<T, B> Reducer for SketchReducer<T, B>
where
    T: PolarsPhysicalType,
    B: ToOwned + ?Sized + 'static,
    B::Owned: fmt::Debug
        + Clone
        + TotalOrd
        + Send
        + Sync
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static,
    for<'a> T::Physical<'a>: Borrow<B>,
{
    type Dtype = T;
    type Value = Sketch<B::Owned>;

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
            a.update(Borrow::<B>::borrow(&b));
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<T>, _seq_id: u64) {
        for value in ca.iter().flatten() {
            v.update(Borrow::<B>::borrow(&value));
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
