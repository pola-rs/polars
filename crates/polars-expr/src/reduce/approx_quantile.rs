use std::borrow::Borrow;
use std::fmt;
use std::marker::PhantomData;

use polars_compute::approx_quantile::{ApproxQuantileMethod, FinalizedSketch, Sketch};
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
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&dtype, |T, B| {
        let reducer = SketchReducer::<T, B>::new(method, error);
        Box::new(VecGroupedReduction::new(dtype.clone(), reducer))
    }))
}

/// Merge the sketches of [`new_approx_quantile_sketch_reduction`] over values of `values_dtype`.
pub fn new_approx_quantile_merge_reduction(
    values_dtype: DataType,
) -> PolarsResult<Box<dyn GroupedReduction>> {
    Ok(with_match_sketch_item!(&values_dtype, |_T, B| {
        let reducer = SketchMergeReducer::<<B as ToOwned>::Owned>(PhantomData);
        Box::new(VecGroupedReduction::new(DataType::Binary, reducer))
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
        let sketches: Vec<_> = v.into_iter().map(Sketch::finalize).collect();
        sketches_to_series(&sketches)
    }
}

struct SketchMergeReducer<I>(PhantomData<I>);

impl<I> Clone for SketchMergeReducer<I> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<I> SketchMergeReducer<I>
where
    I: fmt::Debug + Clone + TotalOrd + serde::de::DeserializeOwned,
{
    fn push_blob(&self, acc: &mut PolarsResult<Vec<FinalizedSketch<I>>>, blob: &[u8]) {
        let Ok(parts) = acc else {
            return;
        };
        match pl_serialize::deserialize_from_reader::<FinalizedSketch<I>, _, false>(blob) {
            Ok(part) => parts.push(part),
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
            self.push_blob(a, b);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &BinaryChunked, _seq_id: u64) {
        for blob in ca.iter().flatten() {
            self.push_blob(v, blob);
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

#[cfg(test)]
mod tests {
    use super::*;

    fn sketch(method: &ApproxQuantileMethod, offset: usize) -> Column {
        let values: Vec<f64> = (0..20_000)
            .map(|i| ((i * 7919 + offset) % 20_000) as f64)
            .collect();
        let mut r =
            new_approx_quantile_sketch_reduction(DataType::Float64, method.clone(), 0.01).unwrap();
        r.resize(1);
        r.update_group(&[&Column::new(PlSmallStr::EMPTY, values)], 0, 0)
            .unwrap();
        r.finalize().unwrap().into_column()
    }

    fn merge(sketches: &[&Column]) -> Box<dyn GroupedReduction> {
        let mut r = new_approx_quantile_merge_reduction(DataType::Float64).unwrap();
        r.resize(1);
        for (seq_id, s) in sketches.iter().enumerate() {
            r.update_group(&[s], 0, seq_id as u64).unwrap();
        }
        r
    }

    /// The merged sketch does not depend on the order or grouping of its parts.
    #[test]
    fn merge_is_order_independent() {
        for method in [
            ApproxQuantileMethod::KLL,
            ApproxQuantileMethod::DoubleReqSketch,
        ] {
            let [a, b, c] = [0, 1, 2].map(|i| sketch(&method, i * 13));
            let expected = merge(&[&a, &b, &c]).finalize().unwrap();
            let reordered = merge(&[&c, &a, &b]).finalize().unwrap();
            assert!(reordered.equals(&expected));

            let mut left = merge(&[&b]);
            let right = merge(&[&c, &a]);
            unsafe { left.combine_subset(&*right, &[0], &[0]).unwrap() };
            assert!(left.finalize().unwrap().equals(&expected));
        }
    }
}
