use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

use polars_compute::approx_quantile::{ApproxQuantileMethod, FinalizedSketch, Sketch};
use polars_core::with_match_physical_numeric_polars_type;
use polars_ops::series::{sketch_dtype, sketch_repr, sketches_to_series};
use polars_utils::pl_serialize;
use polars_utils::total_ord::TotalOrd;

use super::*;

/// Runs `$body` with `$T` the type of [`sketch_repr`] for `$dtype` and `$B` the
/// borrowed type of the sketch items.
macro_rules! with_match_sketch_item {
    ($dtype:expr, | $T:ident, $B:ident | $body:expr) => {{
        let dtype: &DataType = $dtype;
        match sketch_dtype(dtype) {
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
            DataType::Binary => {
                type $T = BinaryType;
                type $B = [u8];
                $body
            },
            DataType::BinaryOffset => {
                type $T = BinaryOffsetType;
                type $B = [u8];
                $body
            },
            physical if physical.is_primitive_numeric() => {
                with_match_physical_numeric_polars_type!(physical, |$P| {
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
    failure_probability: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    new_sketch_reduction(dtype, method, error, failure_probability, true)
}

/// Like [`new_approx_quantile_sketch_reduction`], but outputs the serialized
/// ingesting state of each group.
pub fn new_approx_quantile_state_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
    failure_probability: f64,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    new_sketch_reduction(dtype, method, error, failure_probability, false)
}

fn new_sketch_reduction(
    dtype: DataType,
    method: ApproxQuantileMethod,
    error: f64,
    failure_probability: f64,
    finalize: bool,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&dtype, |T, B| {
        let reducer = SketchReducer::<T, B>::new(method, error, failure_probability, finalize);
        Box::new(VecGroupedReduction::new(dtype.clone(), reducer))
    }))
}

/// Unions finalized sketches of disjoint items over values of `values_dtype`
/// into one finalized sketch per group, retaining every item.
pub fn new_approx_quantile_union_reduction(
    values_dtype: DataType,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&values_dtype, |_T, B| {
        let reducer = SketchUnionReducer::<<B as ToOwned>::Owned>(PhantomData);
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
        failure_probability: f64,
        finalize: bool,
    ) -> Self {
        Self {
            template: Sketch::new(&method, error, failure_probability),
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
        sketch_repr(s).unwrap()
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
struct SketchUnionReducer<I>(PhantomData<I>);

impl<I> Reducer for SketchUnionReducer<I>
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
    type Value = PolarsResult<Vec<FinalizedSketch<I>>>;

    fn init(&self) -> Self::Value {
        Ok(Vec::new())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        match (a, b) {
            (Ok(a), Ok(b)) => a.extend_from_slice(b),
            (a @ Ok(_), Err(e)) => *a = Err(e.clone()),
            (Err(_), _) => {},
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, _seq_id: u64) {
        if let Some(b) = b {
            push_blob(a, b);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &BinaryChunked, _seq_id: u64) {
        for blob in ca.iter().flatten() {
            push_blob(v, blob);
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
            .map(|parts| parts.map(FinalizedSketch::union))
            .collect::<PolarsResult<Vec<_>>>()?;
        sketches_to_series(&sketches)
    }
}

fn push_blob<I>(acc: &mut PolarsResult<Vec<FinalizedSketch<I>>>, blob: &[u8])
where
    I: fmt::Debug + Clone + TotalOrd + serde::de::DeserializeOwned,
{
    let Ok(parts) = acc else {
        return;
    };
    match pl_serialize::deserialize_from_reader::<FinalizedSketch<I>, _, false>(blob) {
        Ok(part) => parts.push(part),
        Err(e) => *acc = Err(e),
    }
}
