use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::series::sketches_to_series;
use polars_utils::pl_serialize;
use polars_utils::total_ord::TotalOrd;

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
    use_formal_bound: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    new_sketch_reduction(dtype, method, error, use_formal_bound, true)
}

/// Like [`new_approx_quantile_sketch_reduction`], but outputs the serialized
/// ingesting state of each group, to be merged by
/// [`new_approx_quantile_merge_reduction`].
pub fn new_approx_quantile_state_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
    use_formal_bound: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    new_sketch_reduction(dtype, method, error, use_formal_bound, false)
}

fn new_sketch_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
    use_formal_bound: bool,
    finalize: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&dtype, |T, B| {
        let reducer = SketchReducer::<T, B>::new(method, error, use_formal_bound, finalize);
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
    use_formal_bound: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&values_dtype, |_T, B| {
        let reducer = SketchMergeReducer::<<B as ToOwned>::Owned> {
            template: Sketch::new(&method, error, use_formal_bound),
        };
        Box::new(VecGroupedReduction::new(DataType::Binary, reducer))
    }))
}

struct SketchReducer<T, B: ToOwned + ?Sized>
where
    B::Owned: fmt::Debug + Clone + TotalOrd,
{
    template: Sketch<B::Owned>,
    /// Output finalized sketches rather than ingesting states.
    finalize: bool,
    dtype: PhantomData<T>,
}

impl<T, B: ToOwned + ?Sized> SketchReducer<T, B>
where
    B::Owned: fmt::Debug + Clone + TotalOrd,
{
    fn new(
        method: ApproxQuantileMethod,
        error: f64,
        use_formal_bound: bool,
        finalize: bool,
    ) -> Self {
        Self {
            template: Sketch::new(&method, error, use_formal_bound),
            finalize,
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
            finalize: self.finalize,
            dtype: PhantomData,
        }
    }
}

impl<T, B> Reducer for SketchReducer<T, B>
where
    T: PolarsPhysicalType,
    B: ToOwned + ?Sized + 'static,
    B::Owned: fmt::Debug + Clone + TotalOrd + Send + Sync + serde::Serialize + 'static,
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
    fn merge_blob(&self, acc: &mut PolarsResult<Sketch<I>>, blob: &[u8]) {
        let Ok(sketch) = acc else {
            return;
        };
        match pl_serialize::deserialize_from_reader::<Sketch<I>, _, false>(blob) {
            Ok(state) => sketch.merge(&state),
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
    type Value = PolarsResult<Sketch<I>>;

    fn init(&self) -> Self::Value {
        Ok(self.template.clone())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        match (a, b) {
            (Ok(a), Ok(b)) => a.merge(b),
            (a @ Ok(_), Err(e)) => *a = Err(e.clone()),
            (Err(_), _) => {},
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
            .map(|s| s.map(Sketch::finalize))
            .collect::<PolarsResult<Vec<_>>>()?;
        sketches_to_series(&sketches)
    }
}
