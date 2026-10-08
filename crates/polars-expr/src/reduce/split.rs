//! Reductions split into a stage that serializes per-group states and one that merges them.

#[cfg(feature = "serde")]
use polars_utils::pl_serialize;

use super::*;

#[derive(Clone, Copy)]
pub(super) enum SplitStage {
    Whole,
    #[cfg(feature = "serde")]
    State,
    /// Merges serialized states and finalizes them as `Whole` would.
    #[cfg(feature = "serde")]
    Merge,
}

/// `Serialize + DeserializeOwned` under `serde`, so that `Whole` builds without it.
#[cfg(feature = "serde")]
pub(super) trait SplitState: serde::Serialize + serde::de::DeserializeOwned {}
#[cfg(feature = "serde")]
impl<T: serde::Serialize + serde::de::DeserializeOwned> SplitState for T {}
#[cfg(not(feature = "serde"))]
pub(super) trait SplitState {}
#[cfg(not(feature = "serde"))]
impl<T> SplitState for T {}

/// Builds `stage` of `reducer`'s reduction over values of `dtype`.
pub(super) fn split_reduction<R>(
    dtype: DataType,
    reducer: R,
    stage: SplitStage,
) -> Box<dyn GroupedReduction>
where
    R: Reducer,
    R::Value: SplitState,
{
    match stage {
        SplitStage::Whole => Box::new(VecGroupedReduction::new(dtype, reducer)),
        #[cfg(feature = "serde")]
        SplitStage::State => Box::new(VecGroupedReduction::new(dtype, StateReducer(reducer))),
        #[cfg(feature = "serde")]
        SplitStage::Merge => Box::new(VecGroupedReduction::new(
            DataType::Binary,
            MergeReducer {
                inner: reducer,
                values_dtype: dtype,
            },
        )),
    }
}

/// Reduces as `R` does, but outputs the serialized states.
#[cfg(feature = "serde")]
#[derive(Clone)]
struct StateReducer<R>(R);

#[cfg(feature = "serde")]
impl<R> Reducer for StateReducer<R>
where
    R: Reducer,
    R::Value: serde::Serialize,
{
    type Dtype = R::Dtype;
    type Value = R::Value;
    const ORDER_INDEPENDENT: bool = R::ORDER_INDEPENDENT;

    fn init(&self) -> Self::Value {
        self.0.init()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        self.0.cast_series(s)
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        self.0.combine(a, b)
    }

    #[inline(always)]
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        seq_id: u64,
    ) {
        self.0.reduce_one(a, b, seq_id)
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        self.0.reduce_ca(v, ca, seq_id)
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none());
        let mut builder = BinaryChunkedBuilder::new(PlSmallStr::EMPTY, v.len());
        let mut blob = Vec::new();
        for state in &v {
            blob.clear();
            pl_serialize::serialize_into_writer::<_, _, false>(&mut blob, state)?;
            builder.append_value(&blob);
        }
        Ok(builder.finish().into_series())
    }
}

/// Merges the serialized states of `R` and finalizes them as `R` does for `values_dtype`.
#[cfg(feature = "serde")]
#[derive(Clone)]
struct MergeReducer<R> {
    inner: R,
    values_dtype: DataType,
}

#[cfg(feature = "serde")]
impl<R> MergeReducer<R>
where
    R: Reducer,
    R::Value: serde::de::DeserializeOwned,
{
    fn merge_blob(&self, acc: &mut PolarsResult<R::Value>, blob: &[u8]) {
        let Ok(state) = acc else {
            return;
        };
        // Blobs come from other processes, so a bad one errors rather than panics.
        match pl_serialize::deserialize_from_reader::<R::Value, _, false>(blob) {
            Ok(other) => self.inner.combine(state, &other),
            Err(e) => *acc = Err(e),
        }
    }
}

#[cfg(feature = "serde")]
impl<R> Reducer for MergeReducer<R>
where
    R: Reducer,
    R::Value: serde::de::DeserializeOwned,
{
    type Dtype = BinaryType;
    type Value = PolarsResult<R::Value>;
    const ORDER_INDEPENDENT: bool = R::ORDER_INDEPENDENT;

    fn init(&self) -> Self::Value {
        Ok(self.inner.init())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        match (a, b) {
            (Ok(a), Ok(b)) => self.inner.combine(a, b),
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
        let states = v.into_iter().collect::<PolarsResult<Vec<_>>>()?;
        self.inner.finish(states, m, &self.values_dtype)
    }
}
