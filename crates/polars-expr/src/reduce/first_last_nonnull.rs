use polars_core::frame::row::AnyValueBufferTrusted;
use polars_core::with_match_physical_numeric_polars_type;

use super::first_last::{First, Last, replace_opt_bytes};
use super::*;

pub fn new_first_nonnull_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_nonnull_reduction_with_policy(dtype, First)
}

pub fn new_last_nonnull_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_nonnull_reduction_with_policy(dtype, Last)
}

fn new_nonnull_reduction_with_policy<P: NonNullPolicy + 'static>(
    dtype: DataType,
    policy: P,
) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VecGroupedReduction::new(
            dtype,
            BoolFirstLastNonNullReducer(policy),
        )),
        _ if dtype.is_primitive_numeric()
            || dtype.is_temporal()
            || dtype.is_decimal()
            || dtype.is_categorical() =>
        {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumFirstLastNonNullReducer::<_, $T>(policy, PhantomData)))
            })
        },
        String | Binary => Box::new(VecGroupedReduction::new(
            dtype,
            BinaryFirstLastNonNullReducer(policy),
        )),
        _ => Box::new(GenericFirstLastNonNullGroupedReduction::new(dtype, policy)),
    }
}

enum FirstOrLast {
    First,
    Last,
}
trait NonNullPolicy: Copy + Send + Sync + 'static {
    fn is_first_or_last(self) -> FirstOrLast;
    fn index(self, len: usize) -> usize;
    fn might_replace(self, new: u64, old: u64, seen: bool) -> bool;
}

impl NonNullPolicy for First {
    fn is_first_or_last(self) -> FirstOrLast {
        FirstOrLast::First
    }

    fn index(self, _len: usize) -> usize {
        0
    }

    fn might_replace(self, new: u64, old: u64, seen: bool) -> bool {
        // Subtracting 1 with wrapping leaves all order unchanged, except it
        // makes 0 (no value) the largest possible.
        // If an item has not yet been found, we still might replace, even if we are higher idx.
        !seen || (new.wrapping_sub(1) < old.wrapping_sub(1))
    }
}

impl NonNullPolicy for Last {
    fn is_first_or_last(self) -> FirstOrLast {
        FirstOrLast::Last
    }

    fn index(self, len: usize) -> usize {
        len - 1
    }

    fn might_replace(self, new: u64, old: u64, seen: bool) -> bool {
        // If an item has not yet been found, we still might replace, even if we are lower idx.
        !seen || (new >= old)
    }
}

struct NumFirstLastNonNullReducer<P: NonNullPolicy, T>(P, PhantomData<T>);

#[derive(Clone, Debug, Default)]
struct ValueForNonNull<T: Clone> {
    value: Option<T>,
    seq: u64,
    seen: bool,
}

impl<P: NonNullPolicy, T> Clone for NumFirstLastNonNullReducer<P, T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<P, T> Reducer for NumFirstLastNonNullReducer<P, T>
where
    P: NonNullPolicy,
    T: PolarsNumericType,
{
    type Dtype = T;
    type Value = ValueForNonNull<T::Native>;

    fn init(&self) -> Self::Value {
        ValueForNonNull::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, old: &mut Self::Value, new: &Self::Value) {
        if new.value.is_some() && self.0.might_replace(new.seq, old.seq, old.seen) {
            old.value = new.value;
            old.seq = new.seq;
            old.seen = true;
        }
    }

    fn reduce_one(&self, old: &mut Self::Value, new: Option<T::Native>, seq_id: u64) {
        if new.is_some() && self.0.might_replace(seq_id, old.seq, old.seen) {
            old.value = new;
            old.seq = seq_id;
            old.seen = true;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if ca.is_empty() {
            return;
        }

        if self.0.might_replace(seq_id, v.seq, v.seen) {
            let val = if ca.has_nulls() {
                match self.0.is_first_or_last() {
                    FirstOrLast::First => ca.first_non_null(),
                    FirstOrLast::Last => ca.last_non_null(),
                }
            } else {
                Some(self.0.index(ca.len()))
            }
            // SAFETY: idx is vlid.
            .and_then(|idx| unsafe { ca.get_unchecked(idx) });

            if val.is_some() {
                v.value = val;
                v.seq = seq_id;
                v.seen = true;
            }
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: ChunkedArray<T> = v
            .into_iter()
            .map(|red_val| red_val.value)
            .collect_ca(PlSmallStr::EMPTY);
        let s = ca.into_series();
        unsafe { s.from_physical_unchecked(dtype) }
    }
}

struct BinaryFirstLastNonNullReducer<P: NonNullPolicy>(P);

impl<P: NonNullPolicy> Clone for BinaryFirstLastNonNullReducer<P> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

impl<P> Reducer for BinaryFirstLastNonNullReducer<P>
where
    P: NonNullPolicy,
{
    type Dtype = BinaryType;
    type Value = ValueForNonNull<Vec<u8>>;

    fn init(&self) -> Self::Value {
        ValueForNonNull::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(&self, old: &mut Self::Value, new: &Self::Value) {
        if new.value.is_some() && self.0.might_replace(new.seq, old.seq, old.seen) {
            old.value.clone_from(&new.value);
            old.seq = new.seq;
            old.seen = true;
        }
    }

    fn reduce_one(&self, old: &mut Self::Value, new: Option<&[u8]>, seq_id: u64) {
        if new.is_some() && self.0.might_replace(seq_id, old.seq, old.seen) {
            replace_opt_bytes(&mut old.value, new);
            old.seq = seq_id;
            old.seen = true;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if ca.is_empty() {
            return;
        }
        if self.0.might_replace(seq_id, v.seq, v.seen) {
            let val = if ca.has_nulls() {
                match self.0.is_first_or_last() {
                    FirstOrLast::First => ca.first_non_null(),
                    FirstOrLast::Last => ca.last_non_null(),
                }
            } else {
                Some(self.0.index(ca.len()))
            }
            .and_then(|idx| ca.get(idx));

            if val.is_some() {
                replace_opt_bytes(&mut v.value, val);
                v.seq = seq_id;
                v.seen = true;
            }
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BinaryChunked = v
            .into_iter()
            .map(|ValueForNonNull { value, .. }| value)
            .collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

#[derive(Clone)]
struct BoolFirstLastNonNullReducer<P: NonNullPolicy>(P);

impl<P> Reducer for BoolFirstLastNonNullReducer<P>
where
    P: NonNullPolicy,
{
    type Dtype = BooleanType;
    type Value = ValueForNonNull<bool>;

    fn init(&self) -> Self::Value {
        ValueForNonNull::default()
    }

    fn combine(&self, old: &mut Self::Value, new: &Self::Value) {
        if new.value.is_some() && self.0.might_replace(new.seq, old.seq, old.seen) {
            old.value = new.value;
            old.seq = new.seq;
            old.seen = new.seen;
        }
    }

    fn reduce_one(&self, old: &mut Self::Value, new: Option<bool>, seq_id: u64) {
        if new.is_some() && self.0.might_replace(seq_id, old.seq, old.seen) {
            old.value = new;
            old.seq = seq_id;
            old.seen = true;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if ca.is_empty() {
            return;
        }
        if self.0.might_replace(seq_id, v.seq, v.seen) {
            let val = if ca.has_nulls() {
                match self.0.is_first_or_last() {
                    FirstOrLast::First => ca.first_non_null(),
                    FirstOrLast::Last => ca.last_non_null(),
                }
            } else {
                Some(self.0.index(ca.len()))
            }
            .and_then(|idx| ca.get(idx));

            if val.is_some() {
                v.value = val;
                v.seq = seq_id;
                v.seen = true;
            }
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BooleanChunked = v
            .into_iter()
            .map(|ValueForNonNull { value, .. }| value)
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

struct GenericFirstLastNonNullGroupedReduction<P: NonNullPolicy> {
    in_dtype: DataType,
    policy: P,
    values: Vec<AnyValue<'static>>,
    seqs: Vec<u64>,
    seen: MutableBitmap,
    evicted_values: Vec<AnyValue<'static>>,
    evicted_seqs: Vec<u64>,
    evicted_seen: BitmapBuilder,
}

impl<P: NonNullPolicy> GenericFirstLastNonNullGroupedReduction<P> {
    fn new(in_dtype: DataType, policy: P) -> Self {
        Self {
            in_dtype,
            policy,
            values: Vec::new(),
            seqs: Vec::new(),
            seen: MutableBitmap::new(),
            evicted_values: Vec::new(),
            evicted_seqs: Vec::new(),
            evicted_seen: BitmapBuilder::new(),
        }
    }
}

impl<P: NonNullPolicy + 'static> GroupedReduction for GenericFirstLastNonNullGroupedReduction<P> {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone(), self.policy))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.seqs.reserve(additional);
        self.seen.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, AnyValue::Null);
        self.seqs.resize(num_groups as usize, 0);
        self.seen.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        if !values.is_empty() {
            let seq_id = seq_id + 1; // We use 0 for 'no value'.
            if self.policy.might_replace(
                seq_id,
                self.seqs[group_idx as usize],
                self.seen.get(group_idx as usize),
            ) {
                let val = if values.has_nulls() {
                    match self.policy.is_first_or_last() {
                        FirstOrLast::First => values
                            .as_materialized_series_maintain_scalar()
                            .first_non_null()
                            .into_value(),
                        FirstOrLast::Last => values
                            .as_materialized_series_maintain_scalar()
                            .last_non_null()
                            .into_value(),
                    }
                } else {
                    // SAFETY: index is valid.
                    unsafe { values.get_unchecked(self.policy.index(values.len())) }
                }
                .into_static();

                if !val.is_null() {
                    self.values[group_idx as usize] = val;
                    self.seqs[group_idx as usize] = seq_id;
                    self.seen.set(group_idx as usize, true);
                }
            }
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &[&Column],
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let seq_id = seq_id + 1; // We use 0 for 'no value'.
        for (i, g) in subset.iter().zip(group_idxs) {
            let idx = g.idx();
            let grp_val = self.values.get_unchecked_mut(idx);
            let grp_seq = self.seqs.get_unchecked_mut(idx);
            if g.should_evict() {
                self.evicted_values
                    .push(core::mem::replace(grp_val, AnyValue::Null));
                self.evicted_seqs.push(core::mem::replace(grp_seq, 0));
                self.evicted_seen.push(self.seen.get_unchecked(idx));
                self.seen.set_unchecked(idx, false);
            }
            if self
                .policy
                .might_replace(seq_id, *grp_seq, self.seen.get_unchecked(idx))
            {
                let val = values.get_unchecked(*i as usize).into_static();
                if !val.is_null() {
                    *grp_val = values.get_unchecked(*i as usize).into_static();
                    *grp_seq = seq_id;
                    self.seen.set_unchecked(idx, true);
                }
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
        assert!(self.in_dtype == other.in_dtype);
        assert!(subset.len() == group_idxs.len());
        for (i, g) in group_idxs.iter().enumerate() {
            let si = *subset.get_unchecked(i) as usize;
            if self.policy.might_replace(
                *other.seqs.get_unchecked(si),
                *self.seqs.get_unchecked(*g as usize),
                self.seen.get_unchecked(*g as usize),
            ) {
                let val = other.values.get_unchecked(si);
                if !val.is_null() {
                    *self.values.get_unchecked_mut(*g as usize) = val.clone();
                    *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(si);
                    self.seen.set_unchecked(*g as usize, true);
                }
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            in_dtype: self.in_dtype.clone(),
            policy: self.policy,
            values: core::mem::take(&mut self.evicted_values),
            seqs: core::mem::take(&mut self.evicted_seqs),
            seen: core::mem::take(&mut self.evicted_seen).into_mut(),
            evicted_values: Vec::new(),
            evicted_seqs: Vec::new(),
            evicted_seen: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        self.seqs.clear();
        let phys_type = self.in_dtype.to_physical();
        let mut buf = AnyValueBufferTrusted::new(&phys_type, self.values.len());
        for v in core::mem::take(&mut self.values) {
            // SAFETY: v is cast to physical.
            unsafe { buf.add_unchecked_owned_physical(&v.to_physical()) };
        }
        // SAFETY: dtype is valid for series.
        unsafe { buf.into_series().from_physical_unchecked(&self.in_dtype) }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
