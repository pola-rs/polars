//! Reductions split into a stage that outputs per-group states and one that merges them.

#[cfg(feature = "serde")]
use polars_arrow::array::{BinaryViewArray, FixedSizeListArray, StructArray};
#[cfg(feature = "approx_quantile")]
use polars_compute::approx_quantile::Sketch;
#[cfg(feature = "serde")]
use polars_compute::moment::{CovState, KurtosisState, PearsonState, SkewState, VarState};
#[cfg(feature = "serde")]
use polars_utils::cardinality_sketch::CardinalitySketch;
#[cfg(feature = "approx_quantile")]
use polars_utils::pl_serialize;

use super::*;

#[derive(Clone, Copy)]
pub(super) enum SplitStage {
    Whole,
    #[cfg(feature = "serde")]
    State,
    /// Merges states and finalizes them as `Whole` would.
    #[cfg(feature = "serde")]
    Merge,
}

/// A per-group state that travels between the stages as a column of `state_dtype()`.
#[cfg(feature = "serde")]
pub(super) trait SplitState: Sized {
    type Decoder<'a>: StateDecoder<Self>
    where
        Self: 'a;
    fn state_dtype() -> DataType;
    fn encode(states: Vec<Self>) -> PolarsResult<Series>;
    /// Errors if `s` is not a single chunk of `state_dtype()`.
    fn decoder(s: &Series) -> PolarsResult<Self::Decoder<'_>>;
}
#[cfg(not(feature = "serde"))]
pub(super) trait SplitState {}
#[cfg(not(feature = "serde"))]
impl<T> SplitState for T {}

/// Reads single states of a state column. States come from other processes, so a malformed one
/// errors rather than panics.
#[cfg(feature = "serde")]
pub(super) trait StateDecoder<S> {
    /// `None` for a null state.
    fn get(&self, row: usize) -> PolarsResult<Option<S>>;
}

#[cfg(feature = "serde")]
fn check_state_column(s: &Series, dtype: &DataType) -> PolarsResult<()> {
    polars_ensure!(
        s.dtype() == dtype,
        SchemaMismatch: "expected a state column of dtype `{dtype}`, got `{}`", s.dtype()
    );
    polars_ensure!(
        s.n_chunks() == 1,
        ComputeError: "expected a state column with one chunk, got {}", s.n_chunks()
    );
    Ok(())
}

/// A state stored as a `Struct` of `N` numeric fields.
#[cfg(feature = "serde")]
pub(super) trait StructState<const N: usize>: Sized {
    type Field: PolarsNumericType;
    const FIELD_NAMES: [&'static str; N];
    fn to_parts(&self) -> [<Self::Field as PolarsNumericType>::Native; N];
    fn from_parts(parts: [<Self::Field as PolarsNumericType>::Native; N]) -> Self;
}

#[cfg(feature = "serde")]
fn struct_state_dtype<S: StructState<N>, const N: usize>() -> DataType {
    DataType::Struct(
        S::FIELD_NAMES
            .iter()
            .map(|name| Field::new((*name).into(), S::Field::get_static_dtype()))
            .collect(),
    )
}

#[cfg(feature = "serde")]
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

#[cfg(feature = "serde")]
pub(super) struct StructDecoder<'a, S: StructState<N>, const N: usize> {
    arr: &'a StructArray,
    fields: [&'a PrimitiveArray<<S::Field as PolarsNumericType>::Native>; N],
    has_null_fields: bool,
}

#[cfg(feature = "serde")]
impl<'a, S: StructState<N>, const N: usize> StructDecoder<'a, S, N> {
    fn new(s: &'a Series) -> PolarsResult<Self> {
        check_state_column(s, &struct_state_dtype::<S, N>())?;
        let arr = s.struct_()?.downcast_as_array();
        let fields: [&PrimitiveArray<_>; N] = std::array::from_fn(|i| {
            arr.values()[i]
                .as_any()
                .downcast_ref::<PrimitiveArray<_>>()
                .unwrap()
        });
        let has_null_fields = fields.iter().any(|f| f.has_nulls());
        Ok(Self {
            arr,
            fields,
            has_null_fields,
        })
    }
}

#[cfg(feature = "serde")]
impl<S: StructState<N>, const N: usize> StateDecoder<S> for StructDecoder<'_, S, N> {
    fn get(&self, row: usize) -> PolarsResult<Option<S>> {
        if self.arr.is_null(row) {
            return Ok(None);
        }
        polars_ensure!(
            !self.has_null_fields || self.fields.iter().all(|f| f.is_valid(row)),
            ComputeError: "invalid state: a valid row has a null field"
        );
        Ok(Some(S::from_parts(std::array::from_fn(|i| {
            self.fields[i].value(row)
        }))))
    }
}

#[cfg(feature = "serde")]
macro_rules! impl_struct_split_state {
    ($ty:ty, $n:literal) => {
        impl SplitState for $ty {
            type Decoder<'a> = StructDecoder<'a, Self, $n>;

            fn state_dtype() -> DataType {
                struct_state_dtype::<Self, $n>()
            }

            fn encode(states: Vec<Self>) -> PolarsResult<Series> {
                encode_struct_states::<Self, $n>(states)
            }

            fn decoder(s: &Series) -> PolarsResult<Self::Decoder<'_>> {
                StructDecoder::new(s)
            }
        }
    };
}

#[cfg(feature = "serde")]
macro_rules! impl_moment_split_state {
    ($ty:ident, $n:literal) => {
        impl StructState<$n> for $ty {
            type Field = Float64Type;
            const FIELD_NAMES: [&'static str; $n] = $ty::FIELD_NAMES;

            fn to_parts(&self) -> [f64; $n] {
                $ty::to_parts(self)
            }

            fn from_parts(parts: [f64; $n]) -> Self {
                $ty::from_parts(parts)
            }
        }

        impl_struct_split_state!($ty, $n);
    };
}

#[cfg(feature = "serde")]
impl_moment_split_state!(VarState, 3);
#[cfg(feature = "serde")]
impl_moment_split_state!(SkewState, 4);
#[cfg(feature = "serde")]
impl_moment_split_state!(KurtosisState, 5);
#[cfg(feature = "serde")]
impl_moment_split_state!(CovState, 4);
#[cfg(feature = "serde")]
impl_moment_split_state!(PearsonState, 6);

/// The `(true_count, count)` state of the boolean variance.
#[cfg(feature = "serde")]
impl StructState<2> for (usize, usize) {
    type Field = UInt64Type;
    const FIELD_NAMES: [&'static str; 2] = ["true_count", "count"];

    fn to_parts(&self) -> [u64; 2] {
        [self.0 as u64, self.1 as u64]
    }

    fn from_parts([true_count, count]: [u64; 2]) -> Self {
        (true_count as usize, count as usize)
    }
}

#[cfg(feature = "serde")]
impl_struct_split_state!((usize, usize), 2);

#[cfg(feature = "serde")]
const CARDINALITY_BUCKETS: usize = 256;

#[cfg(feature = "serde")]
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

    fn decoder(s: &Series) -> PolarsResult<Self::Decoder<'_>> {
        check_state_column(s, &Self::state_dtype())?;
        let arr = s.array()?.downcast_as_array();
        let buckets = arr
            .values()
            .as_any()
            .downcast_ref::<PrimitiveArray<u8>>()
            .unwrap();
        Ok(CardinalityDecoder { arr, buckets })
    }
}

#[cfg(feature = "serde")]
pub(super) struct CardinalityDecoder<'a> {
    arr: &'a FixedSizeListArray,
    buckets: &'a PrimitiveArray<u8>,
}

#[cfg(feature = "serde")]
impl StateDecoder<CardinalitySketch> for CardinalityDecoder<'_> {
    fn get(&self, row: usize) -> PolarsResult<Option<CardinalitySketch>> {
        if self.arr.is_null(row) {
            return Ok(None);
        }
        let start = row * CARDINALITY_BUCKETS;
        polars_ensure!(
            self.buckets
                .validity()
                .is_none_or(|v| v.null_count_range(start, CARDINALITY_BUCKETS) == 0),
            ComputeError: "invalid state: a valid row has a null bucket"
        );
        let buckets = &self.buckets.values()[start..start + CARDINALITY_BUCKETS];
        Ok(Some(CardinalitySketch::from_state(
            buckets.try_into().unwrap(),
        )))
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
        let mut builder = BinaryChunkedBuilder::new(PlSmallStr::EMPTY, states.len());
        let mut blob = Vec::new();
        for state in &states {
            blob.clear();
            pl_serialize::serialize_into_writer::<_, _, false>(&mut blob, state)?;
            builder.append_value(&blob);
        }
        Ok(builder.finish().into_series())
    }

    fn decoder(s: &Series) -> PolarsResult<Self::Decoder<'_>> {
        check_state_column(s, &Self::state_dtype())?;
        Ok(BinaryStateDecoder {
            arr: s.binary()?.downcast_as_array(),
            _state: PhantomData,
        })
    }
}

#[cfg(feature = "approx_quantile")]
pub(super) struct BinaryStateDecoder<'a, S> {
    arr: &'a BinaryViewArray,
    _state: PhantomData<S>,
}

#[cfg(feature = "approx_quantile")]
impl<S: serde::de::DeserializeOwned> StateDecoder<S> for BinaryStateDecoder<'_, S> {
    fn get(&self, row: usize) -> PolarsResult<Option<S>> {
        self.arr
            .get(row)
            .map(pl_serialize::deserialize_from_reader::<S, _, false>)
            .transpose()
    }
}

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
        SplitStage::Merge => Box::new(MergeGroupedReduction::new(ReducerFinish {
            inner: reducer,
            values_dtype: dtype,
        })),
    }
}

/// Reduces as `R` does, but outputs the encoded states.
#[cfg(feature = "serde")]
#[derive(Clone)]
struct StateReducer<R>(R);

#[cfg(feature = "serde")]
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

/// Merges states and finalizes them.
#[cfg(feature = "serde")]
pub(super) trait StateFinish: Clone + Send + Sync + 'static {
    type State: SplitState + Clone + Send + Sync + 'static;
    fn init(&self) -> Self::State;
    fn combine(&self, a: &mut Self::State, b: &Self::State);
    fn finish(&self, states: Vec<Self::State>) -> PolarsResult<Series>;
}

/// Finalizes the states of `R` as `R` does for `values_dtype`.
#[cfg(feature = "serde")]
#[derive(Clone)]
struct ReducerFinish<R> {
    inner: R,
    values_dtype: DataType,
}

#[cfg(feature = "serde")]
impl<R> StateFinish for ReducerFinish<R>
where
    R: Reducer,
    R::Value: SplitState,
{
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
#[cfg(feature = "serde")]
pub(super) struct MergeGroupedReduction<F: StateFinish> {
    values: Vec<F::State>,
    evicted_values: Vec<F::State>,
    finish: F,
}

#[cfg(feature = "serde")]
impl<F: StateFinish> MergeGroupedReduction<F> {
    pub(super) fn new(finish: F) -> Self {
        Self {
            values: Vec::new(),
            evicted_values: Vec::new(),
            finish,
        }
    }
}

#[cfg(feature = "serde")]
fn state_column<'a>(values: &[&'a Column]) -> PolarsResult<&'a Series> {
    let &[values] = values else {
        polars_bail!(ComputeError: "expected one state column, got {}", values.len())
    };
    Ok(values.as_materialized_series())
}

#[cfg(feature = "serde")]
impl<F: StateFinish> GroupedReduction for MergeGroupedReduction<F> {
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
        let states = state_column(values)?.rechunk();
        let decoder = F::State::decoder(&states)?;
        let grp = &mut self.values[group_idx as usize];
        for row in 0..states.len() {
            if let Some(state) = decoder.get(row)? {
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
        assert!(subset.len() == group_idxs.len());
        let decoder = F::State::decoder(state_column(values)?)?;
        for (i, g) in subset.iter().zip(group_idxs) {
            // SAFETY: indices are in-bounds guaranteed by trait.
            let grp = unsafe { self.values.get_unchecked_mut(g.idx()) };
            if g.should_evict() {
                let old = core::mem::replace(grp, self.finish.init());
                self.evicted_values.push(old);
            }
            if let Some(state) = decoder.get(*i as usize)? {
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

// The state and merge stages have no Python entry point, so they are tested here.
#[cfg(all(test, feature = "serde"))]
mod tests {
    use super::super::var_std::*;
    use super::*;

    type Reduction = Box<dyn GroupedReduction>;

    /// Reduces `values[..][i]` into group `groups[i]` of `num_groups` and finalizes.
    fn run(
        r: &dyn GroupedReduction,
        values: &[Series],
        groups: &[IdxSize],
        num_groups: IdxSize,
    ) -> PolarsResult<Series> {
        let mut r = r.new_empty();
        r.resize(num_groups);
        let cols: Vec<Column> = values.iter().cloned().map(Column::from).collect();
        let cols: Vec<&Column> = cols.iter().collect();
        let subset: Vec<IdxSize> = (0..groups.len() as IdxSize).collect();
        unsafe { r.update_groups_subset(&cols, &subset, groups, 0)? };
        r.finalize()
    }

    struct Partition {
        values: Vec<Series>,
        groups: Vec<IdxSize>,
    }

    fn partition(values: Vec<Series>, groups: &[IdxSize]) -> Partition {
        Partition {
            values,
            groups: groups.to_vec(),
        }
    }

    /// Runs `state` on each partition and merges the states with `merge`.
    fn run_split(
        state: &dyn GroupedReduction,
        merge: &dyn GroupedReduction,
        partitions: &[Partition],
        num_groups: IdxSize,
    ) -> PolarsResult<Series> {
        let mut merge = merge.new_empty();
        merge.resize(num_groups);
        let all: Vec<IdxSize> = (0..num_groups).collect();
        for p in partitions {
            let states = Column::from(run(state, &p.values, &p.groups, num_groups)?);
            assert_eq!(states.dtype(), &reduction_output_dtype(state)?);
            unsafe { merge.update_groups_subset(&[&states], &all, &all, 0)? };
        }
        merge.finalize()
    }

    /// Runs `whole` on all partitions at once.
    fn run_whole(
        whole: &dyn GroupedReduction,
        partitions: &[Partition],
        num_groups: IdxSize,
    ) -> PolarsResult<Series> {
        let mut values = partitions[0].values.clone();
        let mut groups = partitions[0].groups.clone();
        for p in &partitions[1..] {
            for (v, pv) in values.iter_mut().zip(&p.values) {
                v.append(pv)?;
            }
            groups.extend_from_slice(&p.groups);
        }
        let values: Vec<Series> = values.iter().map(|v| v.rechunk()).collect();
        run(whole, &values, &groups, num_groups)
    }

    /// Asserts equal up to floating-point rounding.
    fn assert_close(left: &Series, right: &Series) {
        assert_eq!(left.dtype(), right.dtype());
        let left = left.cast(&DataType::Float64).unwrap();
        let right = right.cast(&DataType::Float64).unwrap();
        for (l, r) in left.f64().unwrap().iter().zip(right.f64().unwrap().iter()) {
            match (l, r) {
                (Some(l), Some(r)) => assert!(
                    (l - r).abs() <= 1e-9 * l.abs().max(1.0) || (l.is_nan() && r.is_nan()),
                    "{left:?} != {right:?}"
                ),
                (l, r) => assert_eq!(l, r, "{left:?} != {right:?}"),
            }
        }
        assert_eq!(left.len(), right.len());
    }

    /// The whole, state and merge reductions of one family.
    struct Family {
        whole: Reduction,
        state: Reduction,
        merge: Reduction,
    }

    fn moment_families(dtype: &DataType) -> Vec<Family> {
        let dt = || dtype.clone();
        let mut out = vec![
            Family {
                whole: new_var_std_reduction(dt(), false, 1).unwrap(),
                state: new_var_std_state_reduction(dt()).unwrap(),
                merge: new_var_std_merge_reduction(dt(), false, 1).unwrap(),
            },
            Family {
                whole: new_var_std_reduction(dt(), true, 0).unwrap(),
                state: new_var_std_state_reduction(dt()).unwrap(),
                merge: new_var_std_merge_reduction(dt(), true, 0).unwrap(),
            },
        ];
        #[cfg(feature = "moment")]
        if !dtype.is_bool() {
            use super::super::skew_kurtosis::*;
            out.push(Family {
                whole: new_skew_reduction(dt(), false).unwrap(),
                state: new_skew_state_reduction(dt()).unwrap(),
                merge: new_skew_merge_reduction(dt(), false).unwrap(),
            });
            out.push(Family {
                whole: new_kurtosis_reduction(dt(), true, false).unwrap(),
                state: new_kurtosis_state_reduction(dt()).unwrap(),
                merge: new_kurtosis_merge_reduction(dt(), true, false).unwrap(),
            });
        }
        out
    }

    /// Two partitions over four groups; group 3 gets no values and group 2 a single one.
    fn moment_partitions(dtype: &DataType) -> Vec<Partition> {
        let p1 = Series::new(
            "x".into(),
            [
                Some(1.5),
                Some(2.0),
                None,
                Some(-3.0),
                Some(7.25),
                Some(0.0),
            ],
        );
        let p2 = Series::new("x".into(), [Some(4.0), Some(0.5), Some(10.0), None]);
        let cast = |s: Series| s.cast(dtype).unwrap();
        vec![
            partition(vec![cast(p1)], &[0, 1, 0, 1, 0, 1]),
            partition(vec![cast(p2)], &[1, 0, 2, 3]),
        ]
    }

    #[test]
    fn moment_split_matches_whole() {
        for dtype in [
            DataType::Float64,
            DataType::Float32,
            DataType::Int32,
            DataType::Boolean,
        ] {
            let partitions = moment_partitions(&dtype);
            for f in moment_families(&dtype) {
                let whole = run_whole(&*f.whole, &partitions, 4).unwrap();
                let split = run_split(&*f.state, &*f.merge, &partitions, 4).unwrap();
                assert_close(&split, &whole);
                assert!(reduction_output_dtype(&*f.state).unwrap().is_struct());
                assert_eq!(reduction_output_dtype(&*f.merge).unwrap(), *whole.dtype());
            }
        }
    }

    #[test]
    fn moment_split_of_empty_groups_matches_whole() {
        let dtype = DataType::Float64;
        let empty = vec![partition(vec![Series::new_empty("x".into(), &dtype)], &[])];
        for f in moment_families(&dtype) {
            let whole = run_whole(&*f.whole, &empty, 2).unwrap();
            let split = run_split(&*f.state, &*f.merge, &[], 2).unwrap();
            assert_close(&split, &whole);
            let split = run_split(&*f.state, &*f.merge, &empty, 2).unwrap();
            assert_close(&split, &whole);
        }
    }

    #[test]
    fn moment_split_of_null_input() {
        let dtype = DataType::Null;
        let partitions = vec![partition(
            vec![Series::full_null("x".into(), 3, &dtype)],
            &[0, 1, 0],
        )];
        for f in moment_families(&dtype) {
            assert_eq!(reduction_output_dtype(&*f.state).unwrap(), DataType::Null);
            let whole = run_whole(&*f.whole, &partitions, 2).unwrap();
            let split = run_split(&*f.state, &*f.merge, &partitions, 2).unwrap();
            assert!(split.equals_missing(&whole));
            assert_eq!(split.dtype(), &DataType::Null);
        }
    }

    /// A state column with one state per group of `values`.
    fn var_states(values: &[f64], groups: &[IdxSize], num_groups: IdxSize) -> Column {
        let state = new_var_std_state_reduction(DataType::Float64).unwrap();
        let values = Series::new("x".into(), values);
        Column::from(run(&*state, &[values], groups, num_groups).unwrap())
    }

    fn var_merge() -> Reduction {
        new_var_std_merge_reduction(DataType::Float64, false, 0).unwrap()
    }

    #[test]
    fn merge_reads_sparse_subset() {
        let states = var_states(&[1.0, 2.0, 10.0, 20.0, 40.0, 5.0], &[0, 0, 1, 1, 1, 2], 3);
        let mut merge = var_merge();
        merge.resize(2);
        unsafe {
            merge
                .update_groups_subset(&[&states], &[2, 0], &[1, 0], 0)
                .unwrap()
        };
        let out = merge.finalize().unwrap();
        assert_close(&out, &Series::new("".into(), [0.25, 0.0]));
    }

    #[test]
    fn merge_evicts_on_null_state() {
        let states = var_states(&[1.0, 2.0], &[0, 0], 1);
        let states = Column::from(
            states
                .as_materialized_series()
                .extend_constant(AnyValue::Null, 1)
                .unwrap()
                .rechunk(),
        );
        let mut merge = var_merge();
        merge.resize(1);
        unsafe {
            merge
                .update_groups_while_evicting(&[&states], &[0], &[EvictIdx::new(0, false)], 0)
                .unwrap();
            merge
                .update_groups_while_evicting(&[&states], &[1], &[EvictIdx::new(0, true)], 0)
                .unwrap();
        }
        let evicted = merge.take_evictions().finalize().unwrap();
        assert_close(&evicted, &Series::new("".into(), [0.25]));
        let out = merge.finalize().unwrap();
        assert_close(&out, &Series::new("".into(), [None::<f64>]));
    }

    #[test]
    fn merge_combines_subset() {
        let states = var_states(&[1.0, 2.0, 10.0, 20.0], &[0, 0, 1, 1], 2);
        let mut a = var_merge();
        let mut b = var_merge();
        a.resize(2);
        b.resize(2);
        unsafe {
            a.update_groups_subset(&[&states], &[0], &[0], 0).unwrap();
            b.update_groups_subset(&[&states], &[0, 1], &[0, 1], 0)
                .unwrap();
            // a[0] = {1, 2}, a[1] = {1, 2} + {10, 20}.
            a.combine_subset(&*b, &[1, 0], &[1, 1]).unwrap();
        }
        let out = a.finalize().unwrap();
        let expected = var_states(&[1.0, 2.0, 1.0, 2.0, 10.0, 20.0], &[0, 0, 1, 1, 1, 1], 2);
        let mut merge = var_merge();
        merge.resize(2);
        unsafe {
            merge
                .update_groups_subset(&[&expected], &[0, 1], &[0, 1], 0)
                .unwrap()
        };
        assert_close(&out, &merge.finalize().unwrap());
    }

    /// Merges `states` into one group.
    fn merge_one(merge: &dyn GroupedReduction, states: Series) -> PolarsResult<Series> {
        let groups = vec![0; states.len()];
        run(merge, &[states], &groups, 1)
    }

    #[test]
    fn merge_rejects_bad_states() {
        // Wrong dtype.
        let floats = Series::new("".into(), [1.0, 2.0]);
        assert!(merge_one(&*var_merge(), floats).is_err());

        // A valid row with a null field.
        let field = |name: &str, v: [Option<f64>; 2]| Series::new(name.into(), v);
        let fields = [
            field("weight", [Some(1.0), Some(1.0)]),
            field("mean", [Some(1.0), None]),
            field("dp", [Some(0.0), Some(0.0)]),
        ];
        let states = StructChunked::from_series("".into(), 2, fields.iter()).unwrap();
        let err = merge_one(&*var_merge(), states.clone().into_series()).unwrap_err();
        assert!(err.to_string().contains("null field"), "{err}");

        // The same null field under an outer-null row is skipped.
        let outer = Bitmap::from_iter([true, false]);
        let states = states.with_outer_validity(Some(outer)).into_series();
        let out = merge_one(&*var_merge(), states).unwrap();
        assert_close(&out, &Series::new("".into(), [0.0]));
    }

    #[cfg(feature = "cov")]
    #[test]
    fn cov_corr_split_matches_whole() {
        use super::super::cov::*;

        let x1 = Series::new(
            "x".into(),
            [Some(1.0), Some(2.0), None, Some(4.0), Some(-1.0)],
        );
        let y1 = Series::new(
            "y".into(),
            [Some(2.0), Some(3.5), Some(1.0), Some(9.0), None],
        );
        let x2 = Series::new("x".into(), [Some(3.0), Some(0.5), Some(6.0), Some(7.0)]);
        let y2 = Series::new("y".into(), [Some(1.0), Some(4.0), Some(2.0), Some(7.5)]);
        for (dx, dy) in [
            (DataType::Float64, DataType::Float64),
            (DataType::Int32, DataType::Float32),
            (DataType::Float32, DataType::Float32),
        ] {
            let cast = |s: &Series, dtype: &DataType| s.cast(dtype).unwrap();
            let partitions = [
                partition(vec![cast(&x1, &dx), cast(&y1, &dy)], &[0, 1, 0, 0, 1]),
                partition(vec![cast(&x2, &dx), cast(&y2, &dy)], &[1, 0, 1, 2]),
            ];
            let empty = [partition(
                vec![cast(&x1, &dx).head(Some(0)), cast(&y1, &dy).head(Some(0))],
                &[],
            )];
            let (x, y) = (|| dx.clone(), || dy.clone());
            let families = [
                Family {
                    whole: new_cov_reduction(x(), y(), 1).unwrap(),
                    state: new_cov_state_reduction(x(), y()).unwrap(),
                    merge: new_cov_merge_reduction(x(), y(), 1).unwrap(),
                },
                Family {
                    whole: new_pearson_corr_reduction(x(), y()).unwrap(),
                    state: new_pearson_corr_state_reduction(x(), y()).unwrap(),
                    merge: new_pearson_corr_merge_reduction(x(), y()).unwrap(),
                },
            ];
            for f in families {
                for partitions in [&partitions[..], &empty[..]] {
                    let whole = run_whole(&*f.whole, partitions, 4).unwrap();
                    let split = run_split(&*f.state, &*f.merge, partitions, 4).unwrap();
                    assert_close(&split, &whole);
                }
                assert!(reduction_output_dtype(&*f.state).unwrap().is_struct());
                assert_eq!(
                    reduction_output_dtype(&*f.merge).unwrap(),
                    reduction_output_dtype(&*f.whole).unwrap()
                );
            }
        }
    }

    #[cfg(feature = "approx_unique")]
    #[test]
    fn approx_n_unique_split_matches_whole() {
        use super::super::approx_n_unique::*;

        let int = |v: &[Option<i32>]| Series::new("x".into(), v);
        let null = |n: usize| Series::full_null("x".into(), n, &DataType::Null);
        let cases = [
            // Mixed.
            vec![
                partition(vec![int(&[Some(1), Some(2), None, Some(2)])], &[0, 1, 0, 1]),
                partition(vec![int(&[Some(3), Some(1), Some(2)])], &[1, 0, 2]),
            ],
            // Empty.
            vec![partition(vec![int(&[])], &[])],
            // All null.
            vec![
                partition(vec![int(&[None, None])], &[0, 1]),
                partition(vec![int(&[None])], &[0]),
            ],
            // `Null` dtype.
            vec![
                partition(vec![null(2)], &[0, 0]),
                partition(vec![null(1)], &[1]),
            ],
            vec![partition(vec![null(0)], &[])],
        ];
        let merge = new_approx_n_unique_merge_reduction();
        for partitions in cases {
            let dtype = partitions[0].values[0].dtype().clone();
            let whole = new_approx_n_unique_reduction(dtype.clone()).unwrap();
            let state = new_approx_n_unique_state_reduction(dtype).unwrap();
            assert_eq!(
                reduction_output_dtype(&*state).unwrap(),
                CardinalitySketch::state_dtype()
            );
            let expected = run_whole(&*whole, &partitions, 4).unwrap();
            let split = run_split(&*state, &*merge, &partitions, 4).unwrap();
            assert!(split.equals_missing(&expected), "{split:?} != {expected:?}");
        }
        assert_eq!(reduction_output_dtype(&*merge).unwrap(), IDX_DTYPE);
    }

    #[cfg(feature = "approx_unique")]
    #[test]
    fn approx_n_unique_merge_rejects_null_bucket() {
        use super::super::approx_n_unique::*;

        let mut buckets = vec![Some(0u8); 2 * CARDINALITY_BUCKETS];
        buckets[3] = None;
        let states = |validity: Option<Bitmap>| {
            let ca = ArrayChunked::from_aligned_values(
                "".into(),
                &DataType::UInt8,
                CARDINALITY_BUCKETS,
                vec![PrimitiveArray::from_iter(buckets.iter().copied()).boxed()],
                2,
            );
            ca.with_validity(validity).into_series()
        };
        let merge = new_approx_n_unique_merge_reduction();
        let err = merge_one(&*merge, states(None)).unwrap_err();
        assert!(err.to_string().contains("null bucket"), "{err}");

        let out = merge_one(&*merge, states(Some(Bitmap::from_iter([false, true])))).unwrap();
        assert!(out.equals_missing(&Series::new("".into(), [0 as IdxSize])));
    }

    #[cfg(feature = "approx_quantile")]
    #[test]
    fn approx_quantile_merge_rejects_bad_blob() {
        use polars_compute::approx_quantile::ApproxQuantileMethod;

        use super::super::approx_quantile::new_approx_quantile_merge_reduction;
        let method = ApproxQuantileMethod::KLL;
        let merge = new_approx_quantile_merge_reduction(DataType::Float64, method, 0.01).unwrap();
        let blobs = Series::new("".into(), [&[0xffu8, 0xff, 0xff][..]]);
        assert!(merge_one(&*merge, blobs).is_err());
    }
}
