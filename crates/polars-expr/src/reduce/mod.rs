#![allow(unsafe_op_in_unsafe_fn)]
mod convert;
mod count;
mod first_last;
mod len;
mod mean;
mod min_max;
mod sum;
mod var_std;

use std::any::Any;
use std::borrow::Cow;
use std::marker::PhantomData;

use arrow::array::{Array, PrimitiveArray, StaticArray};
use arrow::bitmap::{Bitmap, BitmapBuilder, MutableBitmap};
pub use convert::into_reduction;
use polars_core::prelude::*;

use crate::EvictIdx;

/// A reduction with groups.
///
/// Each group has its own reduction state that values can be aggregated into.
pub trait GroupedReduction: Any + Send + Sync {
    /// Returns a new empty reduction.
    fn new_empty(&self) -> Box<dyn GroupedReduction>;

    /// Reserves space in this GroupedReduction for an additional number of groups.
    fn reserve(&mut self, additional: usize);

    /// Resizes this GroupedReduction to the given number of groups.
    ///
    /// While not an actual member of the trait, the safety preconditions below
    /// refer to self.num_groups() as given by the last call of this function.
    fn resize(&mut self, num_groups: IdxSize);

    /// Updates the specified group with the given values.
    ///
    /// For order-sensitive grouped reductions, seq_id can be used to resolve
    /// order between calls/multiple reductions.
    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()>;

    /// Updates this GroupedReduction with new values. values[subset[i]] should
    /// be added to reduction self[group_idxs[i]]. For order-sensitive grouped
    /// reductions, seq_id can be used to resolve order between calls/multiple
    /// reductions.
    ///
    /// # Safety
    /// The subset and group_idxs are in-bounds.
    unsafe fn update_groups_subset(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.len() < (1 << (IdxSize::BITS - 1)));
        let evict_group_idxs = core::mem::transmute::<&[IdxSize], &[EvictIdx]>(group_idxs);
        self.update_groups_while_evicting(values, subset, evict_group_idxs, seq_id)
    }

    /// Updates this GroupedReduction with new values. values[subset[i]] should
    /// be added to reduction self[group_idxs[i]]. For order-sensitive grouped
    /// reductions, seq_id can be used to resolve order between calls/multiple
    /// reductions. If the group_idxs[i] has its evict bit set the current value
    /// in the group should be evicted and reset before updating.
    ///
    /// # Safety
    /// The subset and group_idxs are in-bounds.
    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        seq_id: u64,
    ) -> PolarsResult<()>;

    /// Combines this GroupedReduction with another. Group other[subset[i]]
    /// should be combined into group self[group_idxs[i]].
    ///
    /// # Safety
    /// subset[i] < other.num_groups() for all i.
    /// group_idxs[i] < self.num_groups() for all i.
    unsafe fn combine_subset(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()>;

    /// Take the accumulated evicted groups.
    fn take_evictions(&mut self) -> Box<dyn GroupedReduction>;

    /// Returns the finalized value per group as a Series.
    ///
    /// After this operation the number of groups is reset to 0.
    fn finalize(&mut self) -> PolarsResult<Series>;

    /// Returns this GroupedReduction as a dyn Any.
    fn as_any(&self) -> &dyn Any;
}

// Helper traits used in the VecGroupedReduction and VecMaskGroupedReduction to
// reduce code duplication.
pub trait Reducer: Send + Sync + Clone + 'static {
    type Dtype: PolarsDataType<IsLogical = FalseT>;
    type Value: Clone + Send + Sync + 'static;
    fn init(&self) -> Self::Value;
    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Borrowed(s)
    }
    fn combine(&self, a: &mut Self::Value, b: &Self::Value);
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        seq_id: u64,
    );
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, seq_id: u64);
    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series>;
}

pub trait NumericReduction: Send + Sync + 'static {
    type Dtype: PolarsNumericType;
    fn init() -> <Self::Dtype as PolarsNumericType>::Native;
    fn combine(
        a: <Self::Dtype as PolarsNumericType>::Native,
        b: <Self::Dtype as PolarsNumericType>::Native,
    ) -> <Self::Dtype as PolarsNumericType>::Native;
    fn reduce_ca(
        ca: &ChunkedArray<Self::Dtype>,
    ) -> Option<<Self::Dtype as PolarsNumericType>::Native>;
}

struct NumReducer<R: NumericReduction>(PhantomData<R>);
impl<R: NumericReduction> NumReducer<R> {
    fn new() -> Self {
        Self(PhantomData)
    }
}
impl<R: NumericReduction> Clone for NumReducer<R> {
    fn clone(&self) -> Self {
        Self(PhantomData)
    }
}

impl<R: NumericReduction> Reducer for NumReducer<R> {
    type Dtype = <R as NumericReduction>::Dtype;
    type Value = <<R as NumericReduction>::Dtype as PolarsNumericType>::Native;

    #[inline(always)]
    fn init(&self) -> Self::Value {
        <R as NumericReduction>::init()
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        *a = <R as NumericReduction>::combine(*a, *b);
    }

    #[inline(always)]
    fn reduce_one(
        &self,
        a: &mut Self::Value,
        b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>,
        _seq_id: u64,
    ) {
        if let Some(b) = b {
            *a = <R as NumericReduction>::combine(*a, b);
        }
    }

    #[inline(always)]
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>, _seq_id: u64) {
        if let Some(r) = <R as NumericReduction>::reduce_ca(ca) {
            *v = <R as NumericReduction>::combine(*v, r);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        let arr = Box::new(PrimitiveArray::<Self::Value>::from_vec(v).with_validity(m));
        Ok(unsafe { Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![arr], dtype) })
    }
}

pub struct VecGroupedReduction<R: Reducer> {
    values: Vec<R::Value>,
    evicted_values: Vec<R::Value>,
    in_dtype: DataType,
    reducer: R,
}

impl<R: Reducer> VecGroupedReduction<R> {
    fn new(in_dtype: DataType, reducer: R) -> Self {
        Self {
            values: Vec::new(),
            evicted_values: Vec::new(),
            in_dtype,
            reducer,
        }
    }
}

impl<R> GroupedReduction for VecGroupedReduction<R>
where
    R: Reducer,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: Vec::new(),
            evicted_values: Vec::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
        })
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, self.reducer.init());
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        self.reducer
            .reduce_ca(&mut self.values[group_idx as usize], ca, seq_id);
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
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = self.reducer.cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            if values.has_nulls() {
                for (i, g) in subset.iter().zip(group_idxs) {
                    let ov = arr.get_unchecked(*i as usize);
                    let grp = self.values.get_unchecked_mut(g.idx());
                    if g.should_evict() {
                        let old = core::mem::replace(grp, self.reducer.init());
                        self.evicted_values.push(old);
                    }
                    self.reducer.reduce_one(grp, ov, seq_id);
                }
            } else {
                for (i, g) in subset.iter().zip(group_idxs) {
                    let v = arr.value_unchecked(*i as usize);
                    let grp = self.values.get_unchecked_mut(g.idx());
                    if g.should_evict() {
                        let old = core::mem::replace(grp, self.reducer.init());
                        self.evicted_values.push(old);
                    }
                    self.reducer.reduce_one(grp, Some(v), seq_id);
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
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let v = other.values.get_unchecked(*i as usize);
                let grp = self.values.get_unchecked_mut(*g as usize);
                self.reducer.combine(grp, v);
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values),
            evicted_values: Vec::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        self.reducer.finish(v, None, &self.in_dtype)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct VecMaskGroupedReduction<R: Reducer> {
    values: Vec<R::Value>,
    mask: MutableBitmap,
    evicted_values: Vec<R::Value>,
    evicted_mask: BitmapBuilder,
    in_dtype: DataType,
    reducer: R,
}

impl<R: Reducer> VecMaskGroupedReduction<R> {
    fn new(in_dtype: DataType, reducer: R) -> Self {
        Self {
            values: Vec::new(),
            mask: MutableBitmap::new(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            in_dtype,
            reducer,
        }
    }
}

impl<R> GroupedReduction for VecMaskGroupedReduction<R>
where
    R: Reducer,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.in_dtype.clone(), self.reducer.clone()))
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, self.reducer.init());
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        self.reducer
            .reduce_ca(&mut self.values[group_idx as usize], ca, seq_id);
        if ca.len() != ca.null_count() {
            self.mask.set(group_idx as usize, true);
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
        let seq_id = seq_id + 1; // So we can use 0 for 'none yet'.
        let values = values.as_materialized_series(); // @scalar-opt
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                let grp = self.values.get_unchecked_mut(g.idx());
                if g.should_evict() {
                    self.evicted_values
                        .push(core::mem::replace(grp, self.reducer.init()));
                    self.evicted_mask.push(self.mask.get_unchecked(g.idx()));
                    self.mask.set_unchecked(g.idx(), false);
                }
                if let Some(v) = ov {
                    self.reducer.reduce_one(grp, Some(v), seq_id);
                    self.mask.set_unchecked(g.idx(), true);
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
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let o = other.mask.get_unchecked(*i as usize);
                if o {
                    let v = other.values.get_unchecked(*i as usize);
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    self.reducer.combine(grp, v);
                    self.mask.set_unchecked(*g as usize, true);
                }
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values),
            mask: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: Vec::new(),
            evicted_mask: BitmapBuilder::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let m = core::mem::take(&mut self.mask);
        self.reducer.finish(v, Some(m.freeze()), &self.in_dtype)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

struct NullGroupedReduction {
    num_groups: IdxSize,
    num_evictions: IdxSize,
    dtype: DataType,
}

impl NullGroupedReduction {
    fn new(dtype: DataType) -> Self {
        Self {
            num_groups: 0,
            num_evictions: 0,
            dtype,
        }
    }
}

impl GroupedReduction for NullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::new(self.dtype.clone()))
    }

    fn reserve(&mut self, _additional: usize) {}

    fn resize(&mut self, num_groups: IdxSize) {
        self.num_groups = num_groups;
    }

    fn update_group(
        &mut self,
        _values: &Column,
        _group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        _values: &Column,
        _subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        for g in group_idxs {
            self.num_evictions += g.should_evict() as IdxSize;
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        _other: &dyn GroupedReduction,
        _subset: &[IdxSize],
        _group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            num_groups: core::mem::replace(&mut self.num_evictions, 0),
            num_evictions: 0,
            dtype: self.dtype.clone(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        Ok(Series::full_null(
            PlSmallStr::EMPTY,
            core::mem::replace(&mut self.num_groups, 0) as usize,
            &self.dtype,
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
