#![allow(unsafe_op_in_unsafe_fn)]
use std::fmt::Debug;
use std::marker::PhantomData;

use polars_core::frame::row::AnyValueBufferTrusted;
use polars_core::with_match_physical_numeric_polars_type;

use super::*;

pub fn new_first_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy::<First>(dtype)
}

pub fn new_last_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy::<Last>(dtype)
}

pub fn new_single_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    new_reduction_with_policy::<Single>(dtype)
}

fn new_reduction_with_policy<P: Policy + 'static>(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VecGroupedReduction::new(
            dtype,
            BoolFirstLastReducer::<P>(PhantomData),
        )),
        _ if dtype.is_primitive_numeric()
            || dtype.is_temporal()
            || dtype.is_decimal()
            || dtype.is_categorical() =>
        {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumFirstLastReducer::<P, $T>(PhantomData)))
            })
        },
        String | Binary => Box::new(VecGroupedReduction::new(
            dtype,
            BinaryFirstLastReducer::<P>(PhantomData),
        )),
        _ => Box::new(GenericFirstLastGroupedReduction::<P>::new(dtype)),
    }
}

trait Policy: Send + Sync + 'static {
    fn index(len: usize) -> usize;
    fn should_replace(new: u64, old: u64) -> bool;
    fn is_single() -> bool {
        false
    }
}

struct First;
impl Policy for First {
    fn index(_len: usize) -> usize {
        0
    }

    fn should_replace(new: u64, old: u64) -> bool {
        // Subtracting 1 with wrapping leaves all order unchanged, except it
        // makes 0 (no value) the largest possible.
        new.wrapping_sub(1) < old.wrapping_sub(1)
    }
}

struct Last;
impl Policy for Last {
    fn index(len: usize) -> usize {
        len - 1
    }

    fn should_replace(new: u64, old: u64) -> bool {
        new >= old
    }
}

struct Single;
impl Policy for Single {
    fn index(_len: usize) -> usize {
        0
    }

    fn should_replace(_new: u64, old: u64) -> bool {
        old == 0
    }

    fn is_single() -> bool {
        true
    }
}

struct NumFirstLastReducer<P, T>(PhantomData<(P, T)>);

#[derive(Clone, Debug)]
struct Value<T: Clone> {
    value: Option<T>,
    seq: u64,
    count: u64,
}

impl<P, T> Clone for NumFirstLastReducer<P, T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
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
        Value {
            value: None,
            seq: 0,
            count: 0,
        }
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if P::should_replace(b.seq, a.seq) {
            *a = b.clone();
        }
        a.count += b.count;
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, seq_id: u64) {
        if P::should_replace(seq_id, a.seq) {
            a.value = b;
            a.seq = seq_id;
        }
        a.count += b.is_some() as u64;
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && P::should_replace(seq_id, v.seq) {
            let val = ca.get(P::index(ca.len()));
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
        if P::is_single() {
            if v.iter().any(|v| v.count == 0) {
                return Err(polars_err!(ComputeError:
                    "aggregation 'single' expected a single value, got an empty group"
                ));
            }
            if let Some(Value { count: n, .. }) = v.iter().find(|v| v.count > 1) {
                return Err(polars_err!(ComputeError:
                    "aggregation 'single' expected a single value, got a group with {n} values"
                ));
            }
        }
        let ca: ChunkedArray<T> = v
            .into_iter()
            .map(|red_val| red_val.value)
            .collect_ca(PlSmallStr::EMPTY);
        let s = ca.into_series();
        unsafe { s.from_physical_unchecked(dtype) }
    }
}

struct BinaryFirstLastReducer<P>(PhantomData<P>);

impl<P> Clone for BinaryFirstLastReducer<P> {
    fn clone(&self) -> Self {
        Self(PhantomData)
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
    type Value = (Option<Vec<u8>>, u64);

    fn init(&self) -> Self::Value {
        (None, 0)
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if P::should_replace(b.1, a.1) {
            a.0.clone_from(&b.0);
            a.1 = b.1;
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, seq_id: u64) {
        if P::should_replace(seq_id, a.1) {
            replace_opt_bytes(&mut a.0, b);
            a.1 = seq_id;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && P::should_replace(seq_id, v.1) {
            replace_opt_bytes(&mut v.0, ca.get(P::index(ca.len())));
            v.1 = seq_id;
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BinaryChunked = v.into_iter().map(|(x, _s)| x).collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

struct BoolFirstLastReducer<P>(PhantomData<P>);

impl<P> Clone for BoolFirstLastReducer<P> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<P> Reducer for BoolFirstLastReducer<P>
where
    P: Policy,
{
    type Dtype = BooleanType;
    type Value = (Option<bool>, u64);

    fn init(&self) -> Self::Value {
        (None, 0)
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if P::should_replace(b.1, a.1) {
            *a = *b;
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, seq_id: u64) {
        if P::should_replace(seq_id, a.1) {
            a.0 = b;
            a.1 = seq_id;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && P::should_replace(seq_id, v.1) {
            v.0 = ca.get(P::index(ca.len()));
            v.1 = seq_id;
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        _dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BooleanChunked = v.into_iter().map(|(x, _s)| x).collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }
}

pub struct GenericFirstLastGroupedReduction<P> {
    in_dtype: DataType,
    values: Vec<AnyValue<'static>>,
    seqs: Vec<u64>,
    evicted_values: Vec<AnyValue<'static>>,
    evicted_seqs: Vec<u64>,
    policy: PhantomData<fn() -> P>,
}

impl<P> GenericFirstLastGroupedReduction<P> {
    fn new(in_dtype: DataType) -> Self {
        Self {
            in_dtype,
            values: Vec::new(),
            seqs: Vec::new(),
            evicted_values: Vec::new(),
            evicted_seqs: Vec::new(),
            policy: PhantomData,
        }
    }
}

impl<P: Policy + 'static> GroupedReduction for GenericFirstLastGroupedReduction<P> {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.seqs.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, AnyValue::Null);
        self.seqs.resize(num_groups as usize, 0);
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
            if P::should_replace(seq_id, self.seqs[group_idx as usize]) {
                self.values[group_idx as usize] = values.get(P::index(values.len()))?.into_static();
                self.seqs[group_idx as usize] = seq_id;
            }
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
            if g.should_evict() {
                self.evicted_values
                    .push(core::mem::replace(grp_val, AnyValue::Null));
                self.evicted_seqs.push(core::mem::replace(grp_seq, 0));
            }
            if P::should_replace(seq_id, *grp_seq) {
                *grp_val = values.get_unchecked(*i as usize).into_static();
                *grp_seq = seq_id;
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
            if P::should_replace(
                *other.seqs.get_unchecked(si),
                *self.seqs.get_unchecked(*g as usize),
            ) {
                *self.values.get_unchecked_mut(*g as usize) =
                    other.values.get_unchecked(si).clone();
                *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(si);
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            in_dtype: self.in_dtype.clone(),
            values: core::mem::take(&mut self.evicted_values),
            seqs: core::mem::take(&mut self.evicted_seqs),
            evicted_values: Vec::new(),
            evicted_seqs: Vec::new(),
            policy: PhantomData,
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        self.seqs.clear();
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

// #[derive(Clone)]
// struct SingleReducer<R: Reducer> {
//     inner: R,
// }

// #[derive(Clone)]
// struct SingleValue<T> {
//     value: (Option<T>, u64),
//     got_too_many: bool,
// }

// impl<R: Reducer<Value = (Option<T>, u64)>, T: Clone + Send + Sync + 'static> Reducer
//     for SingleReducer<R>
// {
//     type Dtype = R::Dtype;
//     type Value = SingleValue<T>;

//     fn init(&self) -> Self::Value {
//         SingleValue {
//             value: self.inner.init(),
//             got_too_many: false,
//         }
//     }

//     fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
//         self.inner.cast_series(s)
//     }

//     fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
//         if a.got_too_many || b.got_too_many || (a.value.0.is_some() && b.value.0.is_some()) {
//             a.got_too_many = true;
//         } else {
//             self.inner.combine(&mut a.value, &b.value)
//         }
//     }

//     fn reduce_one(
//         &self,
//         a: &mut Self::Value,
//         b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
//         seq_id: u64,
//     ) {
//         if a.got_too_many || (a.value.0.is_some() && b.is_some()) {
//             a.got_too_many = true;
//         } else {
//             self.inner.reduce_one(&mut a.value, b, seq_id)
//         }
//     }

//     fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
//         if v.got_too_many || (v.value.0.is_some() && !ca.is_empty()) {
//             v.got_too_many = true;
//         } else {
//             self.inner.reduce_ca(v, ca, seq_id)
//         }
//     }

//     fn finish(
//         &self,
//         v: Vec<Self::Value>,
//         m: Option<Bitmap>,
//         dtype: &DataType,
//     ) -> PolarsResult<Series> {
//         if v.iter().all(|x| x.value.0.is_none()) {
//             return Err(polars_err!(
//                 ComputeError: "Single reduction got no non-null values in any group [amber]"
//             ));
//         }
//         if v.iter().any(|x| x.got_too_many) {
//             return Err(polars_err!(
//                 ComputeError: "Single reduction got no non-null values in any group [amber]"
//             ));
//         }
//         let v = v.into_iter().map(|x| x.value).collect::<Vec<_>>();
//         self.inner.finish(v, m, dtype)
//     }
// }
