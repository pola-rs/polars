use std::fmt;
use std::marker::PhantomData;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::series::sketches_to_series;
use polars_utils::pl_serialize;
use polars_utils::total_ord::TotalOrd;

use super::*;

/// Runs `$body` with `$T` the physical type reduced for `$dtype` and `$I` the
/// type of the sketch items.
macro_rules! with_match_sketch_item {
    ($dtype:expr, | $T:ident, $I:ident | $body:expr) => {{
        let dtype: &DataType = $dtype;
        match dtype {
            DataType::Boolean => {
                type $T = BooleanType;
                type $I = bool;
                $body
            },
            DataType::String => {
                type $T = StringType;
                type $I = String;
                $body
            },
            _ if dtype.is_primitive_numeric() || dtype.is_temporal() || dtype.is_decimal() => {
                with_match_physical_numeric_polars_type!(dtype.to_physical(), |$P| {
                    type $T = $P;
                    type $I = <$P as PolarsNumericType>::Native;
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
    new_sketch_reduction(dtype, method, error, true)
}

/// Like [`new_approx_quantile_sketch_reduction`], but outputs the serialized
/// ingesting state of each group, to be merged by
/// [`new_approx_quantile_merge_reduction`].
pub fn new_approx_quantile_state_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    new_sketch_reduction(dtype, method, error, false)
}

fn new_sketch_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
    finalize: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&dtype, |T, I| {
        let reducer = SketchReducer::<T, I>::new(method, error, finalize);
        Box::new(VecGroupedReduction::new(dtype.clone(), reducer))
    }))
}

/// Merges the states of [`new_approx_quantile_state_reduction`] over values of
/// `values_dtype`, and outputs the same sketches as
/// [`new_approx_quantile_sketch_reduction`].
pub fn new_approx_quantile_merge_reduction(
    values_dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&values_dtype, |_T, I| {
        let reducer = SketchMergeReducer::<I> {
            template: Sketch::new(&method, error),
        };
        Box::new(VecGroupedReduction::new(DataType::Binary, reducer))
    }))
}

struct SketchReducer<T, I: fmt::Debug + Clone + TotalOrd> {
    template: Sketch<I>,
    /// Output finalized sketches rather than ingesting states.
    finalize: bool,
    dtype: PhantomData<T>,
}

impl<T, I: fmt::Debug + Clone + TotalOrd> SketchReducer<T, I> {
    fn new(method: ApproxQuantileMethod, error: f64, finalize: bool) -> Self {
        Self {
            template: Sketch::new(&method, error),
            finalize,
            dtype: PhantomData,
        }
    }
}

impl<T, I: fmt::Debug + Clone + TotalOrd> Clone for SketchReducer<T, I> {
    fn clone(&self) -> Self {
        Self {
            template: self.template.clone(),
            finalize: self.finalize,
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
        if !self.finalize {
            return sketches_to_series(&v);
        }
        let sketches: Vec<_> = v.into_iter().map(Sketch::finalize).collect();
        sketches_to_series(&sketches)
    }
}

#[derive(Clone)]
struct SketchMergeReducer<I: fmt::Debug + Clone + TotalOrd> {
    template: Sketch<I>,
}

impl<I> SketchMergeReducer<I>
where
    I: fmt::Debug + Clone + TotalOrd + serde::de::DeserializeOwned,
{
    fn merge_blob(&self, acc: &mut PolarsResult<Option<Sketch<I>>>, blob: &[u8]) {
        let Ok(sketch) = acc else {
            return;
        };
        match pl_serialize::deserialize_from_reader::<Sketch<I>, _, false>(blob) {
            Ok(state) => match sketch {
                Some(sketch) => sketch.merge(&state),
                None => *sketch = Some(state),
            },
            Err(e) => *acc = Err(e),
        }
    }
}

impl<I> Reducer for SketchMergeReducer<I>
where
    I: fmt::Debug
        + Clone
        + TotalOrd
        + Send
        + Sync
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static,
{
    type Dtype = BinaryType;
    /// `None` until the group's first state, which is adopted rather than
    /// merged into an empty sketch.
    type Value = PolarsResult<Option<Sketch<I>>>;

    fn init(&self) -> Self::Value {
        Ok(None)
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        match (a, b) {
            (Ok(Some(a)), Ok(Some(b))) => a.merge(b),
            (Ok(a @ None), Ok(b)) => *a = b.clone(),
            (Ok(Some(_)), Ok(None)) | (Err(_), _) => {},
            (a @ Ok(_), Err(e)) => *a = Err(e.clone()),
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, _seq_id: u64) {
        if let Some(b) = b {
            self.merge_blob(a, b);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &BinaryChunked, _seq_id: u64) {
        for blob in ca.iter().flatten() {
            self.merge_blob(v, blob);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let sketches = v
            .into_iter()
            .map(|s| Ok(s?.unwrap_or_else(|| self.template.clone()).finalize()))
            .collect::<PolarsResult<Vec<_>>>()?;
        sketches_to_series(&sketches)
    }
}
