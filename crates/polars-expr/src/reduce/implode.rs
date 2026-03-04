#![allow(unsafe_op_in_unsafe_fn)]
use std::fmt::Debug;
use std::marker::PhantomData;

use arrow::array::{BinaryArray, BinaryViewArray, BinaryViewArrayGeneric, ListArray, MutableBinaryViewArray, MutableBooleanArray, MutablePrimitiveArray};
use arrow::offset::{Offsets, OffsetsBuffer};
use arrow::pushable::Pushable;
use polars_core::frame::row::AnyValueBufferTrusted;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::UnitVec;

use super::*;

pub fn new_unordered_implode_reduction(
    dtype: DataType,
) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecGroupedReduction as VGR;
    match dtype {
        Boolean => Box::new(VGR::new(dtype, BoolUnorderedImplodeReducer)),
        _ if dtype.is_primitive_numeric()
            || dtype.is_temporal()
            || dtype.is_decimal()
            || dtype.is_categorical()
            || dtype.is_enum() =>
        {
            with_match_physical_numeric_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VGR::new(dtype, NumUnorderedImplodeReducer::<$T>(PhantomData)))
            })
        },
        String | Binary => Box::new(VGR::new(dtype, BinaryUnorderedImplodeReducer)),
        _ => todo!()
    }
}


struct NumUnorderedImplodeReducer<T>(PhantomData<T>);

#[derive(Clone, Default)]
struct UnorderedList<T> {
    values: UnitVec<T>,
    null_count: usize,
}

impl<T> Clone for NumUnorderedImplodeReducer<T> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<T: PolarsNumericType> Reducer for NumUnorderedImplodeReducer<T> {
    type Dtype = T;
    type Value = UnorderedList<T::Native>;

    fn init(&self) -> Self::Value {
        UnorderedList::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.values.extend(b.values.iter().copied());
        a.null_count += b.null_count;
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<T::Native>, _seq_id: u64) {
        if let Some(x) = b {
            a.values.push(x);
        } else {
            a.null_count += 1;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.values.extend(ca.downcast_iter().flat_map(|arr| arr.non_null_values_iter()));
        v.null_count += ca.null_count();
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let total_len = v.iter().map(|l| l.values.len() + l.null_count).sum();
        let mut out = MutablePrimitiveArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for list in v.into_iter() {
            offsets.try_push(list.values.len() + list.null_count).unwrap();
            out.extend_values(list.values.into_iter());
            out.extend_null(list.null_count);
        }

        let values = out.freeze();
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let arr = ListArray::new(list_dtype.to_arrow(CompatLevel::newest()), offsets.freeze(), values.boxed(), None);
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        let s = ca.into_series();
        unsafe { s.from_physical_unchecked(&list_dtype) }
    }
}

#[derive(Clone)]
struct BinaryUnorderedImplodeReducer;

impl Reducer for BinaryUnorderedImplodeReducer {
    type Dtype = BinaryType;
    type Value = UnorderedList<Vec<u8>>;

    fn init(&self) -> Self::Value {
        UnorderedList::default()
    }

    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.values.extend(b.values.iter().cloned());
        a.null_count += b.null_count;
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, _seq_id: u64) {
        if let Some(x) = b {
            a.values.push(x.to_vec());
        } else {
            a.null_count += 1;
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.values.extend(ca.downcast_iter().flat_map(|arr| arr.non_null_values_iter()).map(|x| x.to_vec()));
        v.null_count += ca.null_count();
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let total_len = v.iter().map(|l| l.values.len() + l.null_count).sum();
        let mut out = MutableBinaryViewArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for list in v.into_iter() {
            offsets.try_push(list.values.len() + list.null_count).unwrap();
            out.extend_values(list.values.into_iter());
            out.extend_null(list.null_count);
        }

        let values: BinaryViewArray = out.freeze();
        let arrow_dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(PlSmallStr::EMPTY, ArrowDataType::BinaryView, true)));
        let arr = ListArray::new(arrow_dtype, offsets.freeze(), values.boxed(), None);
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        ca.into_series().cast(&list_dtype)
    }
}


#[derive(Clone, Default)]
struct UnorderedBoolList {
    true_count: usize,
    false_count: usize,
    null_count: usize,
}

#[derive(Clone)]
struct BoolUnorderedImplodeReducer;

impl Reducer for BoolUnorderedImplodeReducer {
    type Dtype = BooleanType;
    type Value = UnorderedBoolList;

    fn init(&self) -> Self::Value {
        UnorderedBoolList::default()
    }
    
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        a.true_count += b.true_count;
        a.false_count += b.false_count;
        a.null_count += b.null_count;
    }
    
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<bool>,
        _seq_id: u64,
    ) {
        match b {
            Some(true) => a.true_count += 1,
            Some(false) => a.false_count += 1,
            None => a.null_count += 1,
        }
    }
    
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        v.true_count += ca.num_trues();
        v.false_count += ca.num_falses();
        v.null_count += ca.null_count();
    }
    
    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.

        let total_len = v.iter().map(|l| l.true_count + l.false_count + l.null_count).sum();
        let mut out = MutableBooleanArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for list in v.into_iter() {
            offsets.try_push(list.true_count + list.false_count + list.null_count).unwrap();
            out.extend_constant(list.true_count, Some(true));
            out.extend_constant(list.false_count, Some(false));
            out.extend_null(list.null_count);
        }
        
        let values = out.freeze();
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let arr = ListArray::new(list_dtype.to_arrow(CompatLevel::newest()), offsets.freeze(), values.boxed(), None);
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        Ok(ca.into_series())
    }
}





// struct GenericFirstLastGroupedReduction<P: Policy> {
//     in_dtype: DataType,
//     policy: P,
//     values: Vec<AnyValue<'static>>,
//     seqs: Vec<u64>,
//     counts: Vec<P::Count>,
//     evicted_values: Vec<AnyValue<'static>>,
//     evicted_seqs: Vec<u64>,
//     evicted_counts: Vec<P::Count>,
// }

// impl<P: Policy> GenericFirstLastGroupedReduction<P> {
//     fn new(in_dtype: DataType, policy: P) -> Self {
//         Self {
//             in_dtype,
//             policy,
//             values: Vec::new(),
//             seqs: Vec::new(),
//             counts: Vec::new(),
//             evicted_values: Vec::new(),
//             evicted_seqs: Vec::new(),
//             evicted_counts: Vec::new(),
//         }
//     }
// }

// impl<P: Policy + 'static> GroupedReduction for GenericFirstLastGroupedReduction<P> {
//     fn new_empty(&self) -> Box<dyn GroupedReduction> {
//         Box::new(Self::new(self.in_dtype.clone(), self.policy))
//     }

//     fn reserve(&mut self, additional: usize) {
//         self.values.reserve(additional);
//         self.seqs.reserve(additional);
//         self.counts.reserve(additional);
//     }

//     fn resize(&mut self, num_groups: IdxSize) {
//         self.values.resize(num_groups as usize, AnyValue::Null);
//         self.seqs.resize(num_groups as usize, 0);
//         self.counts.resize(num_groups as usize, P::Count::default());
//     }

//     fn update_group(
//         &mut self,
//         values: &[&Column],
//         group_idx: IdxSize,
//         seq_id: u64,
//     ) -> PolarsResult<()> {
//         let &[values] = values else { unreachable!() };
//         assert!(values.dtype() == &self.in_dtype);
//         if !values.is_empty() {
//             let seq_id = seq_id + 1; // We use 0 for 'no value'.
//             if self
//                 .policy
//                 .should_replace(seq_id, self.seqs[group_idx as usize])
//             {
//                 self.values[group_idx as usize] =
//                     values.get(self.policy.index(values.len()))?.into_static();
//                 self.seqs[group_idx as usize] = seq_id;
//             }
//             P::add_count(&mut self.counts[group_idx as usize], values.len());
//         }
//         Ok(())
//     }

//     unsafe fn update_groups_while_evicting(
//         &mut self,
//         values: &[&Column],
//         subset: &[IdxSize],
//         group_idxs: &[EvictIdx],
//         seq_id: u64,
//     ) -> PolarsResult<()> {
//         let &[values] = values else { unreachable!() };
//         assert!(values.dtype() == &self.in_dtype);
//         assert!(subset.len() == group_idxs.len());
//         let seq_id = seq_id + 1; // We use 0 for 'no value'.
//         for (i, g) in subset.iter().zip(group_idxs) {
//             let grp_val = self.values.get_unchecked_mut(g.idx());
//             let grp_seq = self.seqs.get_unchecked_mut(g.idx());
//             let grp_count = self.counts.get_unchecked_mut(g.idx());
//             if g.should_evict() {
//                 self.evicted_values
//                     .push(core::mem::replace(grp_val, AnyValue::Null));
//                 self.evicted_seqs.push(core::mem::replace(grp_seq, 0));
//                 self.evicted_counts.push(core::mem::take(grp_count));
//             }
//             if self.policy.should_replace(seq_id, *grp_seq) {
//                 *grp_val = values.get_unchecked(*i as usize).into_static();
//                 *grp_seq = seq_id;
//             }
//             P::add_count(self.counts.get_unchecked_mut(g.idx()), 1);
//         }
//         Ok(())
//     }

//     unsafe fn combine_subset(
//         &mut self,
//         other: &dyn GroupedReduction,
//         subset: &[IdxSize],
//         group_idxs: &[IdxSize],
//     ) -> PolarsResult<()> {
//         let other = other.as_any().downcast_ref::<Self>().unwrap();
//         assert!(self.in_dtype == other.in_dtype);
//         assert!(subset.len() == group_idxs.len());
//         for (i, g) in group_idxs.iter().enumerate() {
//             let si = *subset.get_unchecked(i) as usize;
//             if self.policy.should_replace(
//                 *other.seqs.get_unchecked(si),
//                 *self.seqs.get_unchecked(*g as usize),
//             ) {
//                 *self.values.get_unchecked_mut(*g as usize) =
//                     other.values.get_unchecked(si).clone();
//                 *self.seqs.get_unchecked_mut(*g as usize) = *other.seqs.get_unchecked(si);
//             }
//             P::combine_count(
//                 self.counts.get_unchecked_mut(*g as usize),
//                 other.counts.get_unchecked(si),
//             );
//         }
//         Ok(())
//     }

//     fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
//         Box::new(Self {
//             in_dtype: self.in_dtype.clone(),
//             policy: self.policy,
//             values: core::mem::take(&mut self.evicted_values),
//             seqs: core::mem::take(&mut self.evicted_seqs),
//             counts: core::mem::take(&mut self.evicted_counts),
//             evicted_values: Vec::new(),
//             evicted_seqs: Vec::new(),
//             evicted_counts: Vec::new(),
//         })
//     }

//     fn finalize(&mut self) -> PolarsResult<Series> {
//         self.seqs.clear();
//         if let Some(allow_empty) = self.policy.item_policy() {
//             for count in self.counts.iter() {
//                 P::check_count(*count, allow_empty)?;
//             }
//         }
//         let phys_type = self.in_dtype.to_physical();
//         let mut buf = AnyValueBufferTrusted::new(&phys_type, self.values.len());
//         for v in core::mem::take(&mut self.values) {
//             // SAFETY: v is cast to physical.
//             unsafe { buf.add_unchecked_owned_physical(&v.to_physical()) };
//         }
//         // SAFETY: dtype is valid for series.
//         unsafe { buf.into_series().from_physical_unchecked(&self.in_dtype) }
//     }

//     fn as_any(&self) -> &dyn Any {
//         self
//     }
// }

// fn check_item_count_is_one<T, P: Policy>(
//     values: &[Value<T, P::Count>],
//     allow_empty: bool,
// ) -> PolarsResult<()> {
//     for v in values {
//         P::check_count(v.count, allow_empty)?;
//     }
//     Ok(())
// }
