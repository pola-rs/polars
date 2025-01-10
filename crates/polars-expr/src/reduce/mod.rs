mod convert;
mod len;
mod mean;
mod min_max;
mod partition;
mod sum;
mod var_std;

use std::any::Any;
use std::borrow::Cow;
use std::marker::PhantomData;

use arrow::array::{Array, PrimitiveArray, StaticArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
pub use convert::into_reduction;
use polars_core::prelude::*;

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
    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()>;

    /// Updates this GroupedReduction with new values. values[i] should
    /// be added to reduction self[group_idxs[i]].
    ///
    /// # Safety
    /// group_idxs[i] < self.num_groups() for all i.
    unsafe fn update_groups(&mut self, values: &Series, group_idxs: &[IdxSize])
        -> PolarsResult<()>;

    /// Combines this GroupedReduction with another. Group other[i]
    /// should be combined into group self[group_idxs[i]].
    ///
    /// # Safety
    /// group_idxs[i] < self.num_groups() for all i.
    unsafe fn combine(
        &mut self,
        other: &dyn GroupedReduction,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()>;

    /// Combines this GroupedReduction with another. Group other[subset[i]]
    /// should be combined into group self[group_idxs[i]].
    ///
    /// # Safety
    /// subset[i] < other.num_groups() for all i.
    /// group_idxs[i] < self.num_groups() for all i.
    unsafe fn gather_combine(
        &mut self,
        other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()>;

    /// Partitions this GroupedReduction into several partitions.
    ///
    /// The ith group of this GroupedReduction should becomes the group_idxs[i]
    /// group in partition partition_idxs[i].
    ///
    /// # Safety
    /// partitions_idxs[i] < partition_sizes.len() for all i.
    /// group_idxs[i] < partition_sizes[partition_idxs[i]] for all i.
    /// Each partition p has an associated set of group_idxs, this set contains
    /// 0..partition_size[p] exactly once.
    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>>;

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
    );
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>);
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
    ) {
        if let Some(b) = b {
            *a = <R as NumericReduction>::combine(*a, b);
        }
    }

    #[inline(always)]
    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) {
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
    in_dtype: DataType,
    reducer: R,
}

impl<R: Reducer> VecGroupedReduction<R> {
    fn new(in_dtype: DataType, reducer: R) -> Self {
        Self {
            values: Vec::new(),
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

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        let values = self.reducer.cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        self.reducer
            .reduce_ca(&mut self.values[group_idx as usize], ca);
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        assert!(values.len() == group_idxs.len());
        let values = self.reducer.cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            if values.has_nulls() {
                for (g, ov) in group_idxs.iter().zip(ca.iter()) {
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    self.reducer.reduce_one(grp, ov);
                }
            } else {
                let mut offset = 0;
                for arr in ca.downcast_iter() {
                    let subgroup = &group_idxs[offset..offset + arr.len()];
                    for (g, v) in subgroup.iter().zip(arr.values_iter()) {
                        let grp = self.values.get_unchecked_mut(*g as usize);
                        self.reducer.reduce_one(grp, Some(v));
                    }
                    offset += arr.len();
                }
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
        assert!(self.in_dtype == other.in_dtype);
        assert!(group_idxs.len() == other.values.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(other.values.iter()) {
                let grp = self.values.get_unchecked_mut(*g as usize);
                self.reducer.combine(grp, v);
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

    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>> {
        partition::partition_vec(self.values, partition_sizes, partition_idxs)
            .into_iter()
            .map(|values| {
                Box::new(Self {
                    values,
                    in_dtype: self.in_dtype.clone(),
                    reducer: self.reducer.clone(),
                }) as _
            })
            .collect()
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
    in_dtype: DataType,
    reducer: R,
}

impl<R: Reducer> VecMaskGroupedReduction<R> {
    fn new(in_dtype: DataType, reducer: R) -> Self {
        Self {
            values: Vec::new(),
            mask: MutableBitmap::new(),
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
        Box::new(Self {
            values: Vec::new(),
            mask: MutableBitmap::new(),
            in_dtype: self.in_dtype.clone(),
            reducer: self.reducer.clone(),
        })
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, self.reducer.init());
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &self.in_dtype);
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        self.reducer
            .reduce_ca(&mut self.values[group_idx as usize], ca);
        if ca.len() != ca.null_count() {
            self.mask.set(group_idx as usize, true);
        }
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &self.in_dtype);
        assert!(values.len() == group_idxs.len());
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, ov) in group_idxs.iter().zip(ca.iter()) {
                if let Some(v) = ov {
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    self.reducer.reduce_one(grp, Some(v));
                    self.mask.set_unchecked(*g as usize, true);
                }
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
        assert!(self.in_dtype == other.in_dtype);
        assert!(group_idxs.len() == other.values.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, (v, o)) in group_idxs
                .iter()
                .zip(other.values.iter().zip(other.mask.iter()))
            {
                if o {
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    self.reducer.combine(grp, v);
                    self.mask.set_unchecked(*g as usize, true);
                }
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

    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>> {
        partition::partition_vec_mask(
            self.values,
            &self.mask.freeze(),
            partition_sizes,
            partition_idxs,
        )
        .into_iter()
        .map(|(values, mask)| {
            Box::new(Self {
                values,
                mask: mask.into_mut(),
                in_dtype: self.in_dtype.clone(),
                reducer: self.reducer.clone(),
            }) as _
        })
        .collect()
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
