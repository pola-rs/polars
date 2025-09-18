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
    match &dtype {
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
        Null => Box::new(NullGroupedReduction::default()),
        String | Binary => Box::new(VecGroupedReduction::new(dtype, BinaryMinReducer)),
        _ if dtype.is_integer() || dtype.is_temporal() || dtype.is_enum() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<Min<$T>>::new()))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VMGR::new(dtype, NumReducer::<Min<Int128Type>>::new())),
        #[cfg(feature = "dtype-categorical")]
        Categorical(cats, map) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            Box::new(VMGR::new(dtype.clone(), CatMinReducer::<$C>(map.clone(), PhantomData)))
        }),
        _ => unimplemented!(),
    }
}

pub fn new_max_reduction(dtype: DataType, propagate_nans: bool) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecMaskGroupedReduction as VMGR;
    match &dtype {
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
        Null => Box::new(NullGroupedReduction::default()),
        String | Binary => Box::new(VecGroupedReduction::new(dtype, BinaryMaxReducer)),
        _ if dtype.is_integer() || dtype.is_temporal() || dtype.is_enum() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<Max<$T>>::new()))
            })
        },
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(VMGR::new(dtype, NumReducer::<Max<Int128Type>>::new())),
        #[cfg(feature = "dtype-categorical")]
        Categorical(cats, map) => with_match_categorical_physical_type!(cats.physical(), |$C| {
            Box::new(VMGR::new(dtype.clone(), CatMaxReducer::<$C>(map.clone(), PhantomData)))
        }),
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
        let arr = BooleanArray::from(v.freeze()).with_validity(Some(m.freeze()));
        Ok(Series::from_array(PlSmallStr::EMPTY, arr))
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
        let arr = BooleanArray::from(v.freeze()).with_validity(Some(m.freeze()));
        Ok(Series::from_array(PlSmallStr::EMPTY, arr))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}

#[cfg(feature = "dtype-categorical")]
struct CatMinReducer<T>(Arc<CategoricalMapping>, PhantomData<T>);

#[cfg(feature = "dtype-categorical")]
impl<T> Clone for CatMinReducer<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

#[cfg(feature = "dtype-categorical")]
impl<T: PolarsCategoricalType> Reducer for CatMinReducer<T> {
    type Dtype = T::PolarsPhysical;
    type Value = T::Native;

    fn init(&self) -> Self::Value {
        T::Native::max_value() // Ensures it's invalid, preferring the other value.
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        let Some(b_s) = self.0.cat_to_str(b.as_cat()) else {
            return;
        };
        let Some(a_s) = self.0.cat_to_str(a.as_cat()) else {
            *a = *b;
            return;
        };

        if b_s < a_s {
            *a = *b;
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<Self::Value>, _seq_id: u64) {
        if let Some(b) = b {
            self.combine(a, &b);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<T::PolarsPhysical>, _seq_id: u64) {
        for cat in ca.iter().flatten() {
            self.combine(v, &cat);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        let cat_ids = PrimitiveArray::from_vec(v).with_validity(m);
        let cat_ids = ChunkedArray::from(cat_ids);
        unsafe {
            Ok(
                CategoricalChunked::<T>::from_cats_and_dtype_unchecked(cat_ids, dtype.clone())
                    .into_series(),
            )
        }
    }
}

#[cfg(feature = "dtype-categorical")]
struct CatMaxReducer<T>(Arc<CategoricalMapping>, PhantomData<T>);

#[cfg(feature = "dtype-categorical")]
impl<T> Clone for CatMaxReducer<T> {
    fn clone(&self) -> Self {
        Self(self.0.clone(), PhantomData)
    }
}

#[cfg(feature = "dtype-categorical")]
impl<T: PolarsCategoricalType> Reducer for CatMaxReducer<T> {
    type Dtype = T::PolarsPhysical;
    type Value = T::Native;

    fn init(&self) -> Self::Value {
        T::Native::max_value() // Ensures it's invalid, preferring the other value.
    }

    #[inline(always)]
    fn cast_series<'a>(&self, s: &'a Series) -> Cow<'a, Series> {
        s.to_physical_repr()
    }

    fn combine(&self, a: &mut Self::Value, b: &Self::Value) {
        let Some(b_s) = self.0.cat_to_str(b.as_cat()) else {
            return;
        };
        let Some(a_s) = self.0.cat_to_str(a.as_cat()) else {
            *a = *b;
            return;
        };

        if b_s > a_s {
            *a = *b;
        }
    }

    fn reduce_one(&self, a: &mut Self::Value, b: Option<Self::Value>, _seq_id: u64) {
        if let Some(b) = b {
            self.combine(a, &b);
        }
    }

    fn reduce_ca(&self, v: &mut Self::Value, ca: &ChunkedArray<T::PolarsPhysical>, _seq_id: u64) {
        for cat in ca.iter().flatten() {
            self.combine(v, &cat);
        }
    }

    fn finish(
        &self,
        v: Vec<Self::Value>,
        m: Option<Bitmap>,
        dtype: &DataType,
    ) -> PolarsResult<Series> {
        let cat_ids = PrimitiveArray::from_vec(v).with_validity(m);
        let cat_ids = ChunkedArray::from(cat_ids);
        unsafe {
            Ok(
                CategoricalChunked::<T>::from_cats_and_dtype_unchecked(cat_ids, dtype.clone())
                    .into_series(),
            )
        }
    }
}

#[derive(Default)]
pub struct NullGroupedReduction {
    length: usize,
    num_evictions: usize,
}

impl GroupedReduction for NullGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn reserve(&mut self, _additional: usize) {}

    fn resize(&mut self, num_groups: IdxSize) {
        self.length = num_groups as usize;
    }

    fn update_group(
        &mut self,
        values: &Column,
        _group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Null);

        // no-op
        Ok(())
    }

    unsafe fn update_groups_while_evicting(
        &mut self,
        values: &Column,
        subset: &[IdxSize],
        group_idxs: &[EvictIdx],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &DataType::Null);
        assert!(subset.len() == group_idxs.len());

        for g in group_idxs {
            self.num_evictions += g.should_evict() as usize;
        }
        Ok(())
    }

    unsafe fn combine_subset(
        &mut self,
        _other: &dyn GroupedReduction,
        subset: &[IdxSize],
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        assert!(subset.len() == group_idxs.len());

        // no-op
        Ok(())
    }

    fn take_evictions(&mut self) -> Box<dyn GroupedReduction> {
        let out = Box::new(Self {
            length: self.num_evictions,
            num_evictions: 0,
        });
        self.num_evictions = 0;
        out
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        Ok(Series::full_null(
            PlSmallStr::EMPTY,
            self.length,
            &DataType::Null,
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
