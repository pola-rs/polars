mod convert;
mod len;
mod mean;
mod min_max;
// #[cfg(feature = "propagate_nans")]
// mod nan_min_max;
mod sum;

use std::any::Any;
use std::borrow::Cow;
use std::marker::PhantomData;

use arrow::array::PrimitiveArray;
use arrow::bitmap::{Bitmap, MutableBitmap};
pub use convert::into_reduction;
use polars_core::prelude::*;

/// A reduction with groups.
///
/// Each group has its own reduction state that values can be aggregated into.
pub trait GroupedReduction: Any + Send {
    /// Returns a new empty reduction.
    fn new_empty(&self) -> Box<dyn GroupedReduction>;

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

    /// Returns the finalized value per group as a Series.
    ///
    /// After this operation the number of groups is reset to 0.
    fn finalize(&mut self) -> PolarsResult<Series>;

    /// Returns this GroupedReduction as a dyn Any.
    fn as_any(&self) -> &dyn Any;
}

// Helper traits used in the VecGroupedReduction and VecMaskGroupedReduction to
// reduce code duplication.
pub trait Reducer: Send + Sync + 'static {
    type Dtype: PolarsDataType;
    type Value: Clone + Send + Sync + 'static;
    fn init() -> Self::Value;
    #[inline(always)]
    fn cast_series(s: &Series) -> Cow<'_, Series> {
        Cow::Borrowed(s)
    }
    fn combine(a: &mut Self::Value, b: &Self::Value);
    fn reduce_one(a: &mut Self::Value, b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>);
    fn reduce_ca(v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>);
    fn finish(v: Vec<Self::Value>, m: Option<Bitmap>, dtype: &DataType) -> PolarsResult<Series>;
}

pub trait NumericReducer: Send + Sync + 'static {
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

impl<T: NumericReducer> Reducer for T {
    type Dtype = <T as NumericReducer>::Dtype;
    type Value = <<T as NumericReducer>::Dtype as PolarsNumericType>::Native;

    #[inline(always)]
    fn init() -> Self::Value {
        <Self as NumericReducer>::init()
    }

    #[inline(always)]
    fn cast_series(s: &Series) -> Cow<'_, Series> {
        s.to_physical_repr()
    }

    #[inline(always)]
    fn combine(a: &mut Self::Value, b: &Self::Value) {
        *a = <Self as NumericReducer>::combine(*a, *b);
    }

    #[inline(always)]
    fn reduce_one(a: &mut Self::Value, b: Option<<Self::Dtype as PolarsDataType>::Physical<'_>>) {
        if let Some(b) = b {
            *a = <Self as NumericReducer>::combine(*a, b);
        }
    }

    #[inline(always)]
    fn reduce_ca(v: &mut Self::Value, ca: &ChunkedArray<Self::Dtype>) {
        if let Some(r) = <Self as NumericReducer>::reduce_ca(ca) {
            *v = <Self as NumericReducer>::combine(*v, r);
        }
    }

    fn finish(v: Vec<Self::Value>, m: Option<Bitmap>, dtype: &DataType) -> PolarsResult<Series> {
        let arr = Box::new(PrimitiveArray::<Self::Value>::from_vec(v).with_validity(m));
        Ok(unsafe { Series::from_chunks_and_dtype_unchecked(PlSmallStr::EMPTY, vec![arr], dtype) })
    }
}

pub struct VecGroupedReduction<R: Reducer> {
    values: Vec<R::Value>,
    in_dtype: DataType,
    reducer: PhantomData<R>,
}

impl<R: Reducer> VecGroupedReduction<R> {
    fn new(in_dtype: DataType) -> Self {
        Self {
            values: Vec::new(),
            in_dtype,
            reducer: PhantomData,
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
            reducer: PhantomData,
        })
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, R::init());
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &self.in_dtype);
        let values = R::cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        R::reduce_ca(&mut self.values[group_idx as usize], ca);
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
        let values = R::cast_series(values);
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, ov) in group_idxs.iter().zip(ca.iter()) {
                let grp = self.values.get_unchecked_mut(*g as usize);
                R::reduce_one(grp, ov);
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
        assert!(self.values.len() == other.values.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(other.values.iter()) {
                let grp = self.values.get_unchecked_mut(*g as usize);
                R::combine(grp, v);
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        R::finish(v, None, &self.in_dtype)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

pub struct VecMaskGroupedReduction<R: Reducer> {
    values: Vec<R::Value>,
    mask: MutableBitmap,
    in_dtype: DataType,
    reducer: PhantomData<R>,
}

impl<R: Reducer> VecMaskGroupedReduction<R> {
    fn new(in_dtype: DataType) -> Self {
        Self {
            values: Vec::new(),
            mask: MutableBitmap::new(),
            in_dtype,
            reducer: PhantomData,
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
            reducer: PhantomData,
        })
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, R::init());
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &self.in_dtype);
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<R::Dtype> = values.as_ref().as_ref().as_ref();
        R::reduce_ca(&mut self.values[group_idx as usize], ca);
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
                    R::reduce_one(grp, Some(v));
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
        assert!(self.values.len() == other.values.len());
        assert!(self.mask.len() == other.mask.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, (v, o)) in group_idxs
                .iter()
                .zip(other.values.iter().zip(other.mask.iter()))
            {
                if o {
                    let grp = self.values.get_unchecked_mut(*g as usize);
                    R::combine(grp, v);
                    self.mask.set_unchecked(*g as usize, true);
                }
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let m = core::mem::take(&mut self.mask);
        R::finish(v, Some(m.freeze()), &self.in_dtype)
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
