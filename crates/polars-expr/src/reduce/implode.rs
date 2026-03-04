#![allow(unsafe_op_in_unsafe_fn)]
use std::marker::PhantomData;

use arrow::array::{
    BinaryViewArray, ListArray, MutableBinaryViewArray,
    MutableBooleanArray, MutablePrimitiveArray,
};
use arrow::offset::Offsets;
use arrow::pushable::Pushable;
use polars_core::chunked_array::builder::AnonymousListBuilder;
use polars_core::with_match_physical_numeric_polars_type;
use polars_utils::UnitVec;
use polars_utils::idx_vec::IdxVec;

use super::*;

pub fn new_unordered_implode_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
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
        _ => Box::new(GenericImplodeGroupedReduction::new(dtype)),
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
        v.values.extend(
            ca.downcast_iter()
                .flat_map(|arr| arr.non_null_values_iter()),
        );
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
            offsets
                .try_push(list.values.len() + list.null_count)
                .unwrap();
            out.extend_values(list.values.into_iter());
            out.extend_null(list.null_count);
        }

        let values = out.freeze();
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let arr = ListArray::new(
            list_dtype.to_arrow(CompatLevel::newest()),
            offsets.freeze(),
            values.boxed(),
            None,
        );
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
        v.values.extend(
            ca.downcast_iter()
                .flat_map(|arr| arr.non_null_values_iter())
                .map(|x| x.to_vec()),
        );
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
            offsets
                .try_push(list.values.len() + list.null_count)
                .unwrap();
            out.extend_values(list.values.into_iter());
            out.extend_null(list.null_count);
        }

        let values: BinaryViewArray = out.freeze();
        let arrow_dtype = ArrowDataType::LargeList(Box::new(ArrowField::new(
            PlSmallStr::EMPTY,
            ArrowDataType::BinaryView,
            true,
        )));
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

    fn reduce_one(&self, a: &mut Self::Value, b: Option<bool>, _seq_id: u64) {
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

        let total_len = v
            .iter()
            .map(|l| l.true_count + l.false_count + l.null_count)
            .sum();
        let mut out = MutableBooleanArray::with_capacity(total_len);
        let mut offsets = Offsets::<i64>::with_capacity(v.len());
        for list in v.into_iter() {
            offsets
                .try_push(list.true_count + list.false_count + list.null_count)
                .unwrap();
            out.extend_constant(list.true_count, Some(true));
            out.extend_constant(list.false_count, Some(false));
            out.extend_null(list.null_count);
        }

        let values = out.freeze();
        let list_dtype = DataType::List(Box::new(dtype.clone()));
        let arr = ListArray::new(
            list_dtype.to_arrow(CompatLevel::newest()),
            offsets.freeze(),
            values.boxed(),
            None,
        );
        let ca = ListChunked::with_chunk(PlSmallStr::EMPTY, arr);
        Ok(ca.into_series())
    }
}

struct GenericImplodeGroupedReduction {
    in_dtype: DataType,
    values: Vec<Series>,
    evicted_values: Vec<Series>,
    gather_indices: Vec<IdxVec>,
    active_groups: Vec<usize>,
}

impl GenericImplodeGroupedReduction {
    fn new(in_dtype: DataType) -> Self {
        Self {
            in_dtype,
            values: Vec::new(),
            evicted_values: Vec::new(),
            gather_indices: Vec::new(),
            active_groups: Vec::new(),
        }
    }
}

impl GroupedReduction for GenericImplodeGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize_with(num_groups as usize, || {
            Series::new_empty(PlSmallStr::EMPTY, &self.in_dtype)
        });
        self.gather_indices.resize_with(num_groups as usize, IdxVec::new);
    }

    fn update_group(
        &mut self,
        values: &[&Column],
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        let &[values] = values else { unreachable!() };
        assert!(values.dtype() == &self.in_dtype);
        self.values[group_idx as usize].append(values.as_materialized_series())?;
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
        assert!(values.dtype() == &self.in_dtype);
        assert!(subset.len() == group_idxs.len());
        for (i, g) in subset.iter().zip(group_idxs) {
            let group_gather = self.gather_indices.get_unchecked_mut(g.idx());
            if group_gather.is_empty() {
                self.active_groups.push(g.idx());
            }
            if g.should_evict() {
                let mut evicted = core::mem::replace(
                    self.values.get_unchecked_mut(g.idx()),
                    Series::new_empty(PlSmallStr::EMPTY, &self.in_dtype),
                );
                if !group_gather.is_empty() {
                    let s = values
                        .take_slice_unchecked(group_gather)
                        .take_materialized_series();
                    evicted.append_owned(s)?;
                    group_gather.clear();
                }
                self.evicted_values.push(evicted);
            }
            group_gather.push(*i);
        }

        for g in self.active_groups.drain(..) {
            let group_gather = self.gather_indices.get_unchecked_mut(g);
            let s = values
                .take_slice_unchecked(group_gather)
                .take_materialized_series();
            self.values.get_unchecked_mut(g).append_owned(s)?;
            group_gather.clear();
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
            self.values.get_unchecked_mut(*g as usize).append(other.values.get_unchecked(si))?;
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            in_dtype: self.in_dtype.clone(),
            values: core::mem::take(&mut self.evicted_values),
            evicted_values: Vec::new(),
            gather_indices: Vec::new(),
            active_groups: Vec::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let mut builder = AnonymousListBuilder::new(
            PlSmallStr::EMPTY,
            self.values.len(),
            Some(self.in_dtype.clone()),
        );
        for v in &self.values {
            builder.append_series(v)?;
        }
        let out = builder.finish().into_series();
        self.values.clear();
        Ok(out)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
