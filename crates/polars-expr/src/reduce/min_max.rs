#![allow(unsafe_op_in_unsafe_fn)]
use std::borrow::Cow;
use std::marker::PhantomData;

use arrow::array::BooleanArray;
use arrow::bitmap::Bitmap;
use num_traits::Bounded;
use polars_core::with_match_physical_integer_polars_type;
#[cfg(feature = "propagate_nans")]
use polars_ops::prelude::nan_propagating_aggregate::ca_nan_agg;
use polars_utils::float::IsFloat;
use polars_utils::min_max::MinMax;

use super::*;

pub fn new_min_reduction(dtype: DataType, propagate_nans: bool) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecMaskGroupedReduction as VMGR;
    match dtype {
        Boolean => Box::new(BoolMinGroupedReduction::default()),
        #[cfg(feature = "propagate_nans")]
        Float32 if propagate_nans => {
            Box::new(VMGR::new(dtype, NumReducer::<NanMin<Float32Type>>::new()))
        },
        #[cfg(feature = "propagate_nans")]
        Float64 if propagate_nans => {
            Box::new(VMGR::new(dtype, NumReducer::<NanMin<Float64Type>>::new()))
        },
        Float32 => Box::new(VMGR::new(dtype, NumReducer::<Min<Float32Type>>::new())),
        Float64 => Box::new(VMGR::new(dtype, NumReducer::<Min<Float64Type>>::new())),
        String | Binary => Box::new(VecGroupedReduction::new(dtype, BinaryMinReducer)),
        _ if dtype.is_integer() || dtype.is_temporal() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<Min<$T>>::new()))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VMGR::new(dtype, NumReducer::<Min<Int128Type>>::new())),
        _ => unimplemented!(),
    }
}

pub fn new_max_reduction(dtype: DataType, propagate_nans: bool) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecMaskGroupedReduction as VMGR;
    match dtype {
        Boolean => Box::new(BoolMaxGroupedReduction::default()),
        #[cfg(feature = "propagate_nans")]
        Float32 if propagate_nans => {
            Box::new(VMGR::new(dtype, NumReducer::<NanMax<Float32Type>>::new()))
        },
        #[cfg(feature = "propagate_nans")]
        Float64 if propagate_nans => {
            Box::new(VMGR::new(dtype, NumReducer::<NanMax<Float64Type>>::new()))
        },
        Float32 => Box::new(VMGR::new(dtype, NumReducer::<Max<Float32Type>>::new())),
        Float64 => Box::new(VMGR::new(dtype, NumReducer::<Max<Float64Type>>::new())),
        String | Binary => Box::new(VecGroupedReduction::new(dtype, BinaryMaxReducer)),
        _ if dtype.is_integer() || dtype.is_temporal() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<Max<$T>>::new()))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VMGR::new(dtype, NumReducer::<Max<Int128Type>>::new())),
        _ => unimplemented!(),
    }
}

// These two variants ignore nans.
struct Min<T>(PhantomData<T>);
struct Max<T>(PhantomData<T>);

// These two variants propagate nans.
#[cfg(feature = "propagate_nans")]
struct NanMin<T>(PhantomData<T>);
#[cfg(feature = "propagate_nans")]
struct NanMax<T>(PhantomData<T>);

impl<T> NumericReduction for Min<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        if T::Native::is_float() {
            T::Native::nan_value()
        } else {
            T::Native::max_value()
        }
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        MinMax::min_ignore_nan(a, b)
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<T>) -> Option<T::Native> {
        ChunkAgg::min(ca)
    }
}

impl<T> NumericReduction for Max<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native>,
{
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        if T::Native::is_float() {
            T::Native::nan_value()
        } else {
            T::Native::min_value()
        }
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        MinMax::max_ignore_nan(a, b)
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<T>) -> Option<T::Native> {
        ChunkAgg::max(ca)
    }
}

#[cfg(feature = "propagate_nans")]
impl<T: PolarsFloatType> NumericReduction for NanMin<T> {
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        T::Native::max_value()
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        MinMax::min_propagate_nan(a, b)
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<T>) -> Option<T::Native> {
        ca_nan_agg(ca, MinMax::min_propagate_nan)
    }
}

#[cfg(feature = "propagate_nans")]
impl<T: PolarsFloatType> NumericReduction for NanMax<T> {
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        T::Native::min_value()
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        MinMax::max_propagate_nan(a, b)
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<T>) -> Option<T::Native> {
        ca_nan_agg(ca, MinMax::max_propagate_nan)
    }
}

#[derive(Clone)]
struct BinaryMinReducer;
#[derive(Clone)]
struct BinaryMaxReducer;

impl Reducer for BinaryMinReducer {
    type Dtype = BinaryType;
    type Value = Option<Vec<u8>>; // TODO: evaluate SmallVec<u8>.

    fn init(&self) -> Self::Value {
        None
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        self.reduce_one(a, b.as_deref(), 0)
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, _seq_id: u64) {
        match (a, b) {
            (_, None) => {},
            (l @ None, Some(r)) => *l = Some(r.to_owned()),
            (Some(l), Some(r)) => {
                if l.as_slice() > r {
                    l.clear();
                    l.extend_from_slice(r);
                }
            },
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &BinaryChunked, _seq_id: u64) {
        self.reduce_one(v, ca.min_binary(), 0)
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BinaryChunked = v.into_iter().collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

impl Reducer for BinaryMaxReducer {
    type Dtype = BinaryType;
    type Value = Option<Vec<u8>>; // TODO: evaluate SmallVec<u8>.

    #[inline(always)]
    fn init(&self) -> Self::Value {
        None
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    #[inline(always)]
    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        self.reduce_one(a, b.as_deref(), 0)
    }

    #[inline(always)]
    fn reduce_one(&self, a: &mut Self::Value, b: Option<&[u8]>, _seq_id: u64) {
        match (a, b) {
            (_, None) => {},
            (l @ None, Some(r)) => *l = Some(r.to_owned()),
            (Some(l), Some(r)) => {
                if l.as_slice() < r {
                    l.clear();
                    l.extend_from_slice(r);
                }
            },
        }
    }

    #[inline(always)]
    fn reduce_ca(&self, v: &mut Self::Value, ca: &BinaryChunked, _seq_id: u64) {
        self.reduce_one(v, ca.max_binary(), 0)
    }

    #[inline(always)]
    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BinaryChunked = v.into_iter().collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

#[derive(Default)]
pub struct BoolMinGroupedReduction {
    values: MutableBitmap,
    mask: MutableBitmap,
    evicted_values: BitmapBuilder,
    evicted_mask: BitmapBuilder,
}

impl GroupedReduction for BoolMinGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, true);
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if !ca.all() {
            self.values.set(group_idx as usize, false);
        }
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
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                if g.should_evict() {
                    self.evicted_values.push(self.values.get_unchecked(g.idx()));
                    self.evicted_mask.push(self.mask.get_unchecked(g.idx()));
                    self.values.set_unchecked(g.idx(), ov.unwrap_or(true));
                    self.mask.set_unchecked(g.idx(), ov.is_some());
                } else {
                    self.values.and_pos_unchecked(g.idx(), ov.unwrap_or(true));
                    self.mask.or_pos_unchecked(g.idx(), ov.is_some());
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
        assert!(subset.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.values
                    .and_pos_unchecked(*g as usize, other.values.get_unchecked(*i as usize));
                self.mask
                    .or_pos_unchecked(*g as usize, other.mask.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values).into_mut(),
            mask: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: BitmapBuilder::new(),
            evicted_mask: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let m = core::mem::take(&mut self.mask);
        let arr = BooleanArray::from(v.freeze())
            .with_validity(Some(m.freeze()))
            .boxed();
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![arr],
                &DataType::Boolean,
            )
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[derive(Default)]
pub struct BoolMaxGroupedReduction {
    values: MutableBitmap,
    mask: MutableBitmap,
    evicted_values: BitmapBuilder,
    evicted_mask: BitmapBuilder,
}

impl GroupedReduction for BoolMaxGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, additional: usize) {
        self.values.reserve(additional);
        self.mask.reserve(additional)
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, false);
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(
        &mut self,
        values: &Column,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if ca.any() {
            self.values.set(group_idx as usize, true);
        }
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
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Boolean);
        assert!(subset.len() == group_idxs.len());
        let values = values.as_materialized_series(); // @scalar-opt
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let arr = ca.downcast_as_array();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                let ov = arr.get_unchecked(*i as usize);
                if g.should_evict() {
                    self.evicted_values.push(self.values.get_unchecked(g.idx()));
                    self.evicted_mask.push(self.mask.get_unchecked(g.idx()));
                    self.values.set_unchecked(g.idx(), ov.unwrap_or(false));
                    self.mask.set_unchecked(g.idx(), ov.is_some());
                } else {
                    self.values.or_pos_unchecked(g.idx(), ov.unwrap_or(false));
                    self.mask.or_pos_unchecked(g.idx(), ov.is_some());
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
        assert!(subset.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (i, g) in subset.iter().zip(group_idxs) {
                self.values
                    .or_pos_unchecked(*g as usize, other.values.get_unchecked(*i as usize));
                self.mask
                    .or_pos_unchecked(*g as usize, other.mask.get_unchecked(*i as usize));
            }
        }
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            values: core::mem::take(&mut self.evicted_values).into_mut(),
            mask: core::mem::take(&mut self.evicted_mask).into_mut(),
            evicted_values: BitmapBuilder::new(),
            evicted_mask: BitmapBuilder::new(),
        })
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.values);
        let m = core::mem::take(&mut self.mask);
        let arr = BooleanArray::from(v.freeze())
            .with_validity(Some(m.freeze()))
            .boxed();
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![arr],
                &DataType::Boolean,
            )
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
