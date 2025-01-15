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

fn new_reduction_with_policy<P: Policy + 'static>(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VecGroupedReduction::new(
            dtype,
            BoolFirstLastReducer::<P>(PhantomData),
        )),
        _ if dtype.is_primitive_numeric() || dtype.is_temporal() => {
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

#[expect(dead_code)]
struct Arbitrary;
impl Policy for Arbitrary {
    fn index(_len: usize) -> usize {
        0
    }

    fn should_replace(_new: u64, old: u64) -> bool {
        old == 0
    }
}

struct NumFirstLastReducer<P, T>(PhantomData<(P, T)>);

impl<P, T> Clone for NumFirstLastReducer<P, T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<P, T> Reducer for NumFirstLastReducer<P, T>
where
    P: Policy,
    T: PolarsNumericType,
    ChunkedArray<T>: IntoSeries,
{
    type Dtype = T;
    type Value = (Option<T::Native>, u64);

    fn init(&self) -> Self::Value {
        (None, 0)
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        if P::should_replace(b.1, a.1) {
            *a = *b;
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, seq_id: u64) {
        if P::should_replace(seq_id, a.1) {
            *a = (b, seq_id);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64) {
        if !ca.is_empty() && P::should_replace(seq_id, v.1) {
            let val = ca.get(P::index(ca.len()));
            *v = (val, seq_id);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: ChunkedArray<T> = v.into_iter().map(|(x, _s)| x).collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
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
    policy: PhantomData<fn() -> P>,
}

impl<P> GenericFirstLastGroupedReduction<P> {
    fn new(in_dtype: DataType) -> Self {
        Self {
            in_dtype,
            values: Vec::new(),
            seqs: Vec::new(),
            policy: PhantomData,
        }
    }
}

impl<P: Policy + 'static> GroupedReduction for GenericFirstLastGroupedReduction<P> {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            in_dtype: self.in_dtype.clone(),
            values: Vec::new(),
            seqs: Vec::new(),
            policy: PhantomData,
        })
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
        values: &Series,
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        if values.len() > 0 {
            let seq_id = seq_id + 1; // We use 0 for 'no value'.
            if P::should_replace(seq_id, self.seqs[group_idx as usize]) {
                self.values[group_idx as usize] = values.get(P::index(values.len()))?.into_static();
                self.seqs[group_idx as usize] = seq_id;
            }
        }
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        let seq_id = seq_id + 1; // We use 0 for 'no value'.
        for (i, g) in group_idxs.iter().enumerate() {
            if P::should_replace(seq_id, *self.seqs.get_unchecked(*g as usize)) {
                *self.values.get_unchecked_mut(*g as usize) = values.get_unchecked(i).into_static();
                *self.seqs.get_unchecked_mut(*g as usize) = seq_id;
            }
        }
        Ok(())
    }

    unsafe fn combine(
        &mut self,
        other: &dyn GroupedReduction,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
        for (i, g) in group_idxs.iter().enumerate() {
            if P::should_replace(
                *other.seqs.get_unchecked(i),
                *self.seqs.get_unchecked(*g as usize),
            ) {
                *self.values.get_unchecked_mut(*g as usize) = other.values.get_unchecked(i).clone();
                *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(i);
            }
        }
        Ok(())
    }

    unsafe fn gather_combine(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        let other = other.as_any().downcast_ref::<Self>().unwrap();
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

    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>> {
        let values = partition::partition_vec(self.values, partition_sizes, partition_idxs);
        let seqs = partition::partition_vec(self.seqs, partition_sizes, partition_idxs);
        std::iter::zip(values, seqs)
            .map(|(values, seqs)| {
                Box::new(Self {
                    in_dtype: self.in_dtype.clone(),
                    values,
                    seqs,
                    policy: PhantomData,
                }) as _
            })
            .collect()
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
