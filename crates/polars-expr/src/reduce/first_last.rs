#![allow(unsafe_op_in_unsafe_fn)]
use std::fmt::Debug;
use std::marker::PhantomData;

use polars_core::frame::row::AnyValueBufferTrusted;
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub fn new_first_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy(dtype, First)
}

pub fn new_last_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy(dtype, Last)
}

pub fn new_item_reduction(dtype: DataType, allow_empty: bool) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy(dtype, Item { allow_empty })
}

fn new_reduction_with_policy<P: Policy + 'static>(
    dtype: DataType,
    policy: P,
) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VecGroupedReduction::new(
            dtype,
            BoolFirstLastReducer(policy),
        )),
        _ if dtype.is_primitive_numeric()
            || dtype.is_temporal()
            || dtype.is_decimal()
            || dtype.is_categorical() =>
        {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumFirstLastReducer::<_, $T>(policy, PhantomData)))
            })
        },
        String | Binary => Box::new(VecGroupedReduction::new(
            dtype,
            BinaryFirstLastReducer(policy),
        )),
        _ => Box::new(GenericFirstLastGroupedReduction::new(dtype, policy)),
    }
}

trait Policy: Copy + Send + Sync + 'static {
    fn index(self, len: usize) -> usize;
    fn should_replace(self, new: u64, old: u64) -> bool;
    fn item_policy(self) -> Option<bool> {
        None
    }
}

#[derive(Clone, Copy)]
struct First;
impl Policy for First {
    fn index(self, _len: usize) -> usize {
        0
    }

    fn should_replace(self, new: u64, old: u64) -> bool {
        // Subtracting 1 with wrapping leaves all order unchanged, except it
        // makes 0 (no value) the largest possible.
        new.wrapping_sub(1) < old.wrapping_sub(1)
    }
}

#[derive(Clone, Copy)]
struct Last;
impl Policy for Last {
    fn index(self, len: usize) -> usize {
        len - 1
    }

    fn should_replace(self, new: u64, old: u64) -> bool {
        new >= old
    }
}

#[derive(Clone, Copy)]
struct Item {
    allow_empty: bool,
}
impl Policy for Item {
    fn index(self, _len: usize) -> usize {
        0
    }

    fn should_replace(self, _new: u64, old: u64) -> bool {
        old == 0
    }

    fn item_policy(self) -> Option<bool> {
        Some(self.allow_empty)
    }
}

struct NumFirstLastReducer<P: Policy, T>(P, PhantomData<T>);

#[derive(Clone, Debug, Default)]
struct Value<T: Clone> {
    value: Option<T>,
    seq: u64,
    count: u64,
}

impl<P: Policy, T> Clone for NumFirstLastReducer<P, T> {
    fn clone(&self) -> Self {
        Self(self.0, PhantomData)
    }
}

impl<P, T> Reducer for NumFirstLastReducer<P, T>
where
    P: Policy,
    T: PolarsNumericType,
{
    type Dtype = T;
    type Value = Value<T::Native>;

    fn init(&self) -> Self::Value {
        Value::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if self.0.should_replace(b.seq, a.seq) {
            a.value = b.value;
            a.seq = b.seq;
        }
        a.count += b.count;
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, seq_id: u64) {
        if self.0.should_replace(seq_id, a.seq) {
            a.value = b;
            a.seq = seq_id;
        }
        a.count += b.is_some() as u64;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && self.0.should_replace(seq_id, v.seq) {
            let val = ca.get(self.0.index(ca.len()));
            v.value = val;
            v.seq = seq_id;
        }
        v.count += ca.len() as u64;
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        if let Some(allow_empty) = self.0.item_policy() {
            check_item_count_is_one(&v, allow_empty)?;
        }
        let ca: ChunkedArray<T> = v
            .into_iter()
            .map(|red_val| red_val.value)
            .collect_ca(PlSmallStr::EMPTY);
        let s = ca.into_series();
        unsafe { s.from_physical_unchecked(dtype) }
    }
}

struct BinaryFirstLastReducer<P>(P);

impl<P: Policy> Clone for BinaryFirstLastReducer<P> {
    fn clone(&self) -> Self {
        Self(self.0)
    }
}

fn replace_opt_bytes(l: &mut Option<Vec<u8>>, r: Option<&[u8]>) {
    match (l, r) {
        (Some(l), Some(r)) => {
            l.clear();
            l.extend_from_slice(r);
        },
        (l, r) => *l = r.map(|s| s.to_owned()),
    }
}

impl<P> Reducer for BinaryFirstLastReducer<P>
where
    P: Policy,
{
    type Dtype = BinaryType;
    type Value = Value<Vec<u8>>;

    fn init(&self) -> Self::Value {
        Value::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if self.0.should_replace(b.seq, a.seq) {
            a.value.clone_from(&b.value);
            a.seq = b.seq;
        }
        a.count += b.count;
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, seq_id: u64) {
        if self.0.should_replace(seq_id, a.seq) {
            replace_opt_bytes(&mut a.value, b);
            a.seq = seq_id;
        }
        a.count += b.is_some() as u64;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && self.0.should_replace(seq_id, v.seq) {
            replace_opt_bytes(&mut v.value, ca.get(self.0.index(ca.len())));
            v.seq = seq_id;
        }
        v.count += ca.len() as u64;
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        if let Some(allow_empty) = self.0.item_policy() {
            check_item_count_is_one(&v, allow_empty)?;
        }
        let ca: BinaryChunked = v
            .into_iter()
            .map(|Value { value, .. }| value)
            .collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

#[derive(Clone)]
struct BoolFirstLastReducer<P: Policy>(P);

impl<P> Reducer for BoolFirstLastReducer<P>
where
    P: Policy,
{
    type Dtype = BooleanType;
    type Value = Value<bool>;

    fn init(&self) -> Self::Value {
        Value::default()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if self.0.should_replace(b.seq, a.seq) {
            a.value = b.value;
            a.seq = b.seq;
        }
        a.count += b.count;
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, seq_id: u64) {
        if self.0.should_replace(seq_id, a.seq) {
            a.value = b;
            a.seq = seq_id;
        }
        a.count += b.is_some() as u64;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && self.0.should_replace(seq_id, v.seq) {
            v.value = ca.get(self.0.index(ca.len()));
            v.seq = seq_id;
        }
        v.count += ca.len() as u64;
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        if let Some(allow_empty) = self.0.item_policy() {
            check_item_count_is_one(&v, allow_empty)?;
        }
        let ca: BooleanChunked = v
            .into_iter()
            .map(|Value { value, .. }| value)
            .collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

struct GenericFirstLastGroupedReduction<P: Policy> {
    in_dtype: DataType,
    policy: P,
    values: Vec<AnyValue<'static>>,
    seqs: Vec<u64>,
    counts: Vec<u64>,
    evicted_values: Vec<AnyValue<'static>>,
    evicted_seqs: Vec<u64>,
    evicted_counts: Vec<u64>,
}

impl<P: Policy> GenericFirstLastGroupedReduction<P> {
    fn new(in_dtype: DataType, policy: P) -> Self {
        Self {
            in_dtype,
            policy,
            values: Vec::new(),
            seqs: Vec::new(),
            counts: Vec::new(),
            evicted_values: Vec::new(),
            evicted_seqs: Vec::new(),
            evicted_counts: Vec::new(),
        }
    }
}

impl<P: Policy + 'static> GroupedReduction for GenericFirstLastGroupedReduction<P> {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone(), self.policy))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.seqs.reserve(additional);
        self.counts.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, AnyValue::Null);
        self.seqs.resize(num_groups as usize, 0);
        self.counts.resize(num_groups as usize, 0);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        if !values.is_empty() {
            let seq_id = seq_id + 1; // We use 0 for 'no value'.
            if self
                .policy
                .should_replace(seq_id, self.seqs[group_idx as usize])
            {
                self.values[group_idx as usize] =
                    values.get(self.policy.index(values.len()))?.into_static();
                self.seqs[group_idx as usize] = seq_id;
            }
            self.counts[group_idx as usize] += values.len() as u64;
        }
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        let seq_id = seq_id + 1; // We use 0 for 'no value'.
        for (i, g) in subset.iter().zip(group_idxs) {
            let grp_val = self.values.get_unchecked_mut(g.idx());
            let grp_seq = self.seqs.get_unchecked_mut(g.idx());
            let grp_count = self.counts.get_unchecked_mut(g.idx());
            if g.should_evict() {
                self.evicted_values
                    .push(core::mem::replace(grp_val, AnyValue::Null));
                self.evicted_seqs.push(core::mem::replace(grp_seq, 0));
                self.evicted_counts.push(core::mem::replace(grp_count, 0));
            }
            if self.policy.should_replace(seq_id, *grp_seq) {
                *grp_val = values.get_unchecked(*i as usize).into_static();
                *grp_seq = seq_id;
            }
            *self.counts.get_unchecked_mut(g.idx()) += 1;
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
            if self.policy.should_replace(
                *other.seqs.get_unchecked(si),
                *self.seqs.get_unchecked(*g as usize),
            ) {
                *self.values.get_unchecked_mut(*g as usize) =
                    other.values.get_unchecked(si).clone();
                *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(si);
            }
            *self.counts.get_unchecked_mut(*g as usize) += *other.counts.get_unchecked(si);
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            in_dtype: self.in_dtype.clone(),
            policy: self.policy,
            values: core::mem::take(&mut self.evicted_values),
            seqs: core::mem::take(&mut self.evicted_seqs),
            counts: core::mem::take(&mut self.evicted_counts),
            evicted_values: Vec::new(),
            evicted_seqs: Vec::new(),
            evicted_counts: Vec::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        self.seqs.clear();
        if let Some(allow_empty) = self.policy.item_policy() {
            for count in self.counts.iter() {
                polars_ensure!(
                    (allow_empty && *count == 0) || *count == 1,
                    item_agg_count_not_one = *count,
                    allow_empty = allow_empty
                );
            }
        }
        unsafe {
            let mut buf = AnyValueBufferTrusted::new(&self.in_dtype, self.values.len());
            for v in core::mem::take(&mut self.values) {
                buf.add_unchecked_owned_physical(&v);
            }
            Ok(buf.into_series())
        }
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

fn check_item_count_is_one<T: Clone>(v: &[Value<T>], allow_empty: bool) -> PolarsResult<()> {
    if let Some(Value { count: n, .. }) = v
        .iter()
        .find(|v| !(allow_empty && v.count == 0) && v.count != 1)
    {
        polars_bail!(item_agg_count_not_one = *n, allow_empty = allow_empty);
    }
    Ok(())
}
