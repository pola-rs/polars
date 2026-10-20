//! The state and merge stages, which pass per-group states between them as columns.

#[cfg(feature = "approx_quantile")]
use polars_arrow::array::BinaryViewArray;
use polars_arrow::array::{FixedSizeListArray, StructArray};
#[cfg(feature = "approx_quantile")]
use polars_compute::approx_quantile::Sketch;
use polars_compute::moment::{CovState, KurtosisState, PearsonState, SkewState, VarState};
use polars_utils::cardinality_sketch::CardinalitySketch;
#[cfg(feature = "approx_quantile")]
use polars_utils::pl_serialize;

#[cfg(feature = "dtype-decimal")]
use super::super::mean::DecimalMeanState;
use super::super::mean::MeanState;
use super::super::var_std::BoolCountState;
use super::*;

/// A per-group state that travels between the stages as a column of `state_dtype()`.
pub(crate) trait SplitState: Sized {
    type Decoder<'a>: StateDecoder<Self>
    where
        Self: 'a;
    fn state_dtype() -> DataType;
    fn encode(states: Vec<Self>) -> PolarsResult<Series>;
    /// Panics if `s` is not a single chunk of `state_dtype()`.
    fn decoder(s: &Series) -> Self::Decoder<'_>;
}

pub(crate) trait StateDecoder<S> {
    /// `None` for a null state.
    fn get(&self, row: usize) -> Option<S>;
}

/// A state stored as a `Struct` of `N` numeric fields.
pub(crate) trait StructState<const N: usize>: Sized {
    type Field: PolarsNumericType;
    const FIELD_NAMES: [&'static str; N];
    fn to_parts(&self) -> [<Self::Field as PolarsNumericType>::Native; N];
    fn from_parts(parts: [<Self::Field as PolarsNumericType>::Native; N]) -> Self;
}

fn struct_state_dtype<S: StructState<N>, const N: usize>() -> DataType {
    DataType::Struct(
        S::FIELD_NAMES
            .iter()
            .map(|name| Field::new((*name).into(), S::Field::get_static_dtype()))
            .collect(),
    )
}

fn encode_struct_states<S: StructState<N>, const N: usize>(states: Vec<S>) -> PolarsResult<Series> {
    let mut parts: [Vec<_>; N] = std::array::from_fn(|_| Vec::with_capacity(states.len()));
    for state in &states {
        for (field, part) in parts.iter_mut().zip(state.to_parts()) {
            field.push(part);
        }
    }
    let fields: Vec<Series> = parts
        .into_iter()
        .zip(S::FIELD_NAMES)
        .map(|(values, name)| ChunkedArray::<S::Field>::from_vec(name.into(), values).into_series())
        .collect();
    Ok(StructChunked::from_series(PlSmallStr::EMPTY, states.len(), fields.iter())?.into_series())
}

pub(crate) struct StructDecoder<'a, S: StructState<N>, const N: usize> {
    arr: &'a StructArray,
    fields: [&'a PrimitiveArray<<S::Field as PolarsNumericType>::Native>; N],
}

impl<'a, S: StructState<N>, const N: usize> StructDecoder<'a, S, N> {
    fn new(s: &'a Series) -> Self {
        assert_eq!(s.dtype(), &struct_state_dtype::<S, N>());
        let arr = s.struct_().unwrap().downcast_as_array();
        let fields = std::array::from_fn(|i| {
            arr.values()[i]
                .as_any()
                .downcast_ref::<PrimitiveArray<_>>()
                .unwrap()
        });
        Self { arr, fields }
    }
}

impl<S: StructState<N>, const N: usize> StateDecoder<S> for StructDecoder<'_, S, N> {
    fn get(&self, row: usize) -> Option<S> {
        self.arr
            .is_valid(row)
            .then(|| S::from_parts(std::array::from_fn(|i| self.fields[i].value(row))))
    }
}

macro_rules! impl_struct_split_state {
    ($ty:ident, $field:ty, $n:literal) => {
        impl StructState<$n> for $ty {
            type Field = $field;
            const FIELD_NAMES: [&'static str; $n] = $ty::FIELD_NAMES;

            fn to_parts(&self) -> [<$field as PolarsNumericType>::Native; $n] {
                $ty::to_parts(self)
            }

            fn from_parts(parts: [<$field as PolarsNumericType>::Native; $n]) -> Self {
                $ty::from_parts(parts)
            }
        }

        impl SplitState for $ty {
            type Decoder<'a> = StructDecoder<'a, Self, $n>;

            fn state_dtype() -> DataType {
                struct_state_dtype::<Self, $n>()
            }

            fn encode(states: Vec<Self>) -> PolarsResult<Series> {
                encode_struct_states::<Self, $n>(states)
            }

            fn decoder(s: &Series) -> Self::Decoder<'_> {
                StructDecoder::new(s)
            }
        }
    };
}

impl_struct_split_state!(VarState, Float64Type, 3);
impl_struct_split_state!(SkewState, Float64Type, 4);
impl_struct_split_state!(KurtosisState, Float64Type, 5);
impl_struct_split_state!(CovState, Float64Type, 4);
impl_struct_split_state!(PearsonState, Float64Type, 6);
pub(crate) struct FieldsDecoder<'a, F> {
    arr: &'a StructArray,
    fields: F,
}

/// Implements [`SplitState`] for a struct of numeric fields of different types.
macro_rules! impl_fields_split_state {
    ($ty:ident { $($idx:tt $field:ident: $pt:ty),+ $(,)? }) => {
        impl SplitState for $ty {
            type Decoder<'a> =
                FieldsDecoder<'a, ($(&'a PrimitiveArray<<$pt as PolarsNumericType>::Native>,)+)>;

            fn state_dtype() -> DataType {
                DataType::Struct(vec![
                    $(Field::new(stringify!($field).into(), <$pt>::get_static_dtype())),+
                ])
            }

            fn encode(states: Vec<Self>) -> PolarsResult<Series> {
                let fields = [$(
                    ChunkedArray::<$pt>::from_vec(
                        stringify!($field).into(),
                        states
                            .iter()
                            .map(|s| s.$field as <$pt as PolarsNumericType>::Native)
                            .collect(),
                    )
                    .into_series()
                ),+];
                Ok(StructChunked::from_series(PlSmallStr::EMPTY, states.len(), fields.iter())?
                    .into_series())
            }

            fn decoder(s: &Series) -> Self::Decoder<'_> {
                assert_eq!(s.dtype(), &Self::state_dtype());
                let arr = s.struct_().unwrap().downcast_as_array();
                let fields = ($(
                    arr.values()[$idx]
                        .as_any()
                        .downcast_ref::<PrimitiveArray<<$pt as PolarsNumericType>::Native>>()
                        .unwrap(),
                )+);
                FieldsDecoder { arr, fields }
            }
        }

        impl<'a> StateDecoder<$ty>
            for FieldsDecoder<'a, ($(&'a PrimitiveArray<<$pt as PolarsNumericType>::Native>,)+)>
        {
            fn get(&self, row: usize) -> Option<$ty> {
                self.arr.is_valid(row).then(|| $ty {
                    $($field: self.fields.$idx.value(row) as _),+
                })
            }
        }
    };
}

impl_fields_split_state!(BoolCountState { 0 true_count: UInt64Type, 1 count: UInt64Type });
impl_fields_split_state!(MeanState { 0 sum: Float64Type, 1 count: UInt64Type });
#[cfg(feature = "dtype-decimal")]
impl_fields_split_state!(DecimalMeanState {
    0 sum: Int128Type,
    1 spilled: Float64Type,
    2 count: UInt64Type,
});

const CARDINALITY_BUCKETS: usize = 256;

impl SplitState for CardinalitySketch {
    type Decoder<'a> = CardinalityDecoder<'a>;

    fn state_dtype() -> DataType {
        DataType::Array(Box::new(DataType::UInt8), CARDINALITY_BUCKETS)
    }

    fn encode(states: Vec<Self>) -> PolarsResult<Series> {
        let len = states.len();
        let mut buckets = Vec::with_capacity(len * CARDINALITY_BUCKETS);
        for sketch in states {
            buckets.extend_from_slice(&sketch.into_state());
        }
        let ca = ArrayChunked::from_aligned_values(
            PlSmallStr::EMPTY,
            &DataType::UInt8,
            CARDINALITY_BUCKETS,
            vec![PrimitiveArray::from_vec(buckets).boxed()],
            len,
        );
        Ok(ca.into_series())
    }

    fn decoder(s: &Series) -> Self::Decoder<'_> {
        assert_eq!(s.dtype(), &Self::state_dtype());
        let arr = s.array().unwrap().downcast_as_array();
        let buckets = arr
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<u8>>()
            .unwrap();
        CardinalityDecoder { arr, buckets }
    }
}

pub(crate) struct CardinalityDecoder<'a> {
    arr: &'a FixedSizeListArray,
    buckets: &'a PrimitiveArray<u8>,
}

impl StateDecoder<CardinalitySketch> for CardinalityDecoder<'_> {
    fn get(&self, row: usize) -> Option<CardinalitySketch> {
        self.arr.is_valid(row).then(|| {
            let start = row * CARDINALITY_BUCKETS;
            let buckets = &self.buckets.values()[start..start + CARDINALITY_BUCKETS];
            CardinalitySketch::from_state(buckets.try_into().unwrap())
        })
    }
}

/// Sketches vary in size, so they travel serialized.
#[cfg(feature = "approx_quantile")]
impl<I> SplitState for Sketch<I>
where
    I: std::fmt::Debug
        + Clone
        + polars_utils::total_ord::TotalOrd
        + serde::Serialize
        + serde::de::DeserializeOwned
        + 'static,
{
    type Decoder<'a> = BinaryStateDecoder<'a, Self>;

    fn state_dtype() -> DataType {
        DataType::Binary
    }

    fn encode(states: Vec<Self>) -> PolarsResult<Series> {
        polars_ops::series::sketches_to_series(&states)
    }

    fn decoder(s: &Series) -> Self::Decoder<'_> {
        assert_eq!(s.dtype(), &Self::state_dtype());
        BinaryStateDecoder {
            arr: s.binary().unwrap().downcast_as_array(),
            _state: PhantomData,
        }
    }
}

#[cfg(feature = "approx_quantile")]
pub(crate) struct BinaryStateDecoder<'a, S> {
    arr: &'a BinaryViewArray,
    _state: PhantomData<S>,
}

#[cfg(feature = "approx_quantile")]
impl<S: serde::de::DeserializeOwned> StateDecoder<S> for BinaryStateDecoder<'_, S> {
    fn get(&self, row: usize) -> Option<S> {
        self.arr
            .get(row)
            .map(|blob| pl_serialize::deserialize_from_reader::<S, _, false>(blob).unwrap())
    }
}

/// Reduces as `R` does, but outputs the encoded states.
#[derive(Clone)]
pub(crate) struct StateReducer<R>(pub R);

impl<R> Reducer for StateReducer<R>
where
    R: Reducer,
    R::Value: SplitState,
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
        R::Value::encode(v)
    }
}

/// Combines states as `F` does, but outputs them encoded.
#[cfg(feature = "cov")]
#[derive(Clone)]
pub(crate) struct Encode<F>(pub F);

#[cfg(feature = "cov")]
impl<F> StateFinish for Encode<F>
where
    F: StateFinish<State: SplitState>,
{
    type State = F::State;

    fn init(&self) -> Self::State {
        self.0.init()
    }

    fn combine(&self, a: &mut Self::State, b: &Self::State) {
        self.0.combine(a, b)
    }

    fn finish(&self, states: Vec<Self::State>) -> PolarsResult<Series> {
        F::State::encode(states)
    }
}

/// Finalizes the states of `R` as `R` does for `values_dtype`.
#[derive(Clone)]
pub(crate) struct ReducerFinish<R> {
    pub inner: R,
    pub values_dtype: DataType,
}

impl<R: Reducer> StateFinish for ReducerFinish<R> {
    type State = R::Value;

    fn init(&self) -> Self::State {
        self.inner.init()
    }

    fn combine(&self, a: &mut Self::State, b: &Self::State) {
        self.inner.combine(a, b)
    }

    fn finish(&self, states: Vec<Self::State>) -> PolarsResult<Series> {
        self.inner.finish(states, None, &self.values_dtype)
    }
}

/// Merges a column of encoded states per group and finalizes them with `F`.
pub(crate) struct MergeGroupedReduction<F: StateFinish> {
    values: Vec<F::State>,
    evicted_values: Vec<F::State>,
    finish: F,
}

impl<F: StateFinish> MergeGroupedReduction<F> {
    pub(crate) fn new(finish: F) -> Self {
        Self {
            values: Vec::new(),
            evicted_values: Vec::new(),
            finish,
        }
    }
}

impl<F> GroupedReduction for MergeGroupedReduction<F>
where
    F: StateFinish<State: SplitState>,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.finish.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, self.finish.init());
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        let states = values.as_materialized_series().rechunk();
        let decoder = F::State::decoder(&states);
        let grp = &mut self.values[group_idx as usize];
        for row in 0..states.len() {
            if let Some(state) = decoder.get(row) {
                self.finish.combine(grp, &state);
            }
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(subset.len() == group_idxs.len());
        let decoder = F::State::decoder(values.as_materialized_series());
        for (i, g) in subset.iter().zip(group_idxs) {
            // SAFETY: indices are in-bounds guaranteed by trait.
            let grp = unsafe { self.values.get_unchecked_mut(g.idx()) };
            if g.should_evict() {
                let old = core::mem::replace(grp, self.finish.init());
                self.evicted_values.push(old);
            }
            if let Some(state) = decoder.get(*i as usize) {
                self.finish.combine(grp, &state);
            }
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        assert!(subset.len() == group_idxs.len());
        for (i, g) in subset.iter().zip(group_idxs) {
            // SAFETY: indices are in-bounds guaranteed by trait.
            unsafe {
                let v = other.values.get_unchecked(*i as usize);
                let grp = self.values.get_unchecked_mut(*g as usize);
                self.finish.combine(grp, v);
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values),
            evicted_values: Vec::new(),
            finish: self.finish.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        self.finish.finish(v)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
