use std::borrow::Cow;
use std::marker::PhantomData;

use arrow::array::BooleanArray;
use arrow::bitmap::Bitmap;
use num_traits::Bounded;
use polars_ops::prelude::nan_propagating_aggregate::ca_nan_agg;
use polars_utils::float::IsFloat;
use polars_utils::min_max::MinMax;

use super::*;

pub fn new_min_reduction(dtype: DataType, propagate_nans: bool) -> Box<dyn GroupedReduction> {
    use DataType::*;
    match dtype {
        Boolean => Box::new(BoolMinGroupedReduction::default()),
        UInt8 => Box::new(VecMaskGroupedReduction::<MinReducer<UInt8Type>>::new(dtype)),
        UInt16 => Box::new(VecMaskGroupedReduction::<MinReducer<UInt16Type>>::new(
            dtype,
        )),
        UInt32 => Box::new(VecMaskGroupedReduction::<MinReducer<UInt32Type>>::new(
            dtype,
        )),
        UInt64 => Box::new(VecMaskGroupedReduction::<MinReducer<UInt64Type>>::new(
            dtype,
        )),
        Int8 => Box::new(VecMaskGroupedReduction::<MinReducer<Int8Type>>::new(dtype)),
        Int16 => Box::new(VecMaskGroupedReduction::<MinReducer<Int16Type>>::new(dtype)),
        Int32 => Box::new(VecMaskGroupedReduction::<MinReducer<Int32Type>>::new(dtype)),
        Int64 => Box::new(VecMaskGroupedReduction::<MinReducer<Int64Type>>::new(dtype)),
        Float32 => {
            if propagate_nans {
                Box::new(VecMaskGroupedReduction::<NanMinReducer<Float32Type>>::new(
                    dtype,
                ))
            } else {
                Box::new(VecMaskGroupedReduction::<MinReducer<Float32Type>>::new(
                    dtype,
                ))
            }
        },
        Float64 => {
            if propagate_nans {
                Box::new(VecMaskGroupedReduction::<NanMinReducer<Float64Type>>::new(
                    dtype,
                ))
            } else {
                Box::new(VecMaskGroupedReduction::<MinReducer<Float64Type>>::new(
                    dtype,
                ))
            }
        },
        Decimal(_, _) => Box::new(VecMaskGroupedReduction::<MinReducer<Int128Type>>::new(
            dtype,
        )),
        String | Binary => Box::new(VecGroupedReduction::<BinaryMinReducer>::new(dtype)),
        Date => Box::new(VecMaskGroupedReduction::<MinReducer<Int32Type>>::new(dtype)),
        Datetime(_, _) | Duration(_) | Time => {
            Box::new(VecMaskGroupedReduction::<MinReducer<Int64Type>>::new(dtype))
        },
        _ => unimplemented!(),
    }
}

pub fn new_max_reduction(dtype: DataType, propagate_nans: bool) -> Box<dyn GroupedReduction> {
    use DataType::*;
    match dtype {
        Boolean => Box::new(BoolMaxGroupedReduction::default()),
        UInt8 => Box::new(VecMaskGroupedReduction::<MaxReducer<UInt8Type>>::new(dtype)),
        UInt16 => Box::new(VecMaskGroupedReduction::<MaxReducer<UInt16Type>>::new(
            dtype,
        )),
        UInt32 => Box::new(VecMaskGroupedReduction::<MaxReducer<UInt32Type>>::new(
            dtype,
        )),
        UInt64 => Box::new(VecMaskGroupedReduction::<MaxReducer<UInt64Type>>::new(
            dtype,
        )),
        Int8 => Box::new(VecMaskGroupedReduction::<MaxReducer<Int8Type>>::new(dtype)),
        Int16 => Box::new(VecMaskGroupedReduction::<MaxReducer<Int16Type>>::new(dtype)),
        Int32 => Box::new(VecMaskGroupedReduction::<MaxReducer<Int32Type>>::new(dtype)),
        Int64 => Box::new(VecMaskGroupedReduction::<MaxReducer<Int64Type>>::new(dtype)),
        Float32 => {
            if propagate_nans {
                Box::new(VecMaskGroupedReduction::<NanMaxReducer<Float32Type>>::new(
                    dtype,
                ))
            } else {
                Box::new(VecMaskGroupedReduction::<MaxReducer<Float32Type>>::new(
                    dtype,
                ))
            }
        },
        Float64 => {
            if propagate_nans {
                Box::new(VecMaskGroupedReduction::<NanMaxReducer<Float64Type>>::new(
                    dtype,
                ))
            } else {
                Box::new(VecMaskGroupedReduction::<MaxReducer<Float64Type>>::new(
                    dtype,
                ))
            }
        },
        Decimal(_, _) => Box::new(VecMaskGroupedReduction::<MaxReducer<Int128Type>>::new(
            dtype,
        )),
        String | Binary => Box::new(VecGroupedReduction::<BinaryMaxReducer>::new(dtype)),
        Date => Box::new(VecMaskGroupedReduction::<MaxReducer<Int32Type>>::new(dtype)),
        Datetime(_, _) | Duration(_) | Time => {
            Box::new(VecMaskGroupedReduction::<MaxReducer<Int64Type>>::new(dtype))
        },
        _ => unimplemented!(),
    }
}

// These two variants ignore nans.
struct MinReducer<T>(PhantomData<T>);
struct MaxReducer<T>(PhantomData<T>);

// These two variants propagate nans.
struct NanMinReducer<T>(PhantomData<T>);
struct NanMaxReducer<T>(PhantomData<T>);

impl<T> NumericReducer for MinReducer<T>
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

impl<T> NumericReducer for MaxReducer<T>
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

impl<T: PolarsFloatType> NumericReducer for NanMinReducer<T> {
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

impl<T: PolarsFloatType> NumericReducer for NanMaxReducer<T> {
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

struct BinaryMinReducer;
struct BinaryMaxReducer;

impl Reducer for BinaryMinReducer {
    type Dtype = BinaryType;
    type Value = Option<Vec<u8>>; // TODO: evaluate SmallVec<u8>.

    fn init() -> Self::Value {
        None
    }

    #[inline(always)]
    fn cast_series(s: &Series) -> Cow<'_, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    fn combine(a: &mut Self::Value, b: &Self::Value) {
        Self::reduce_one(a, b.as_deref())
    }

    fn reduce_one(a: &mut Self::Value, b: Option<&[u8]>) {
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

    fn reduce_ca(v: &mut Self::Value, ca: &BinaryChunked) {
        Self::reduce_one(v, ca.min_binary())
    }

    fn finish(v: Vec<Self::Value>, m: Option<Bitmap>, dtype: &DataType) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BinaryChunked = v.into_iter().collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

impl Reducer for BinaryMaxReducer {
    type Dtype = BinaryType;
    type Value = Option<Vec<u8>>; // TODO: evaluate SmallVec<u8>.

    #[inline(always)]
    fn init() -> Self::Value {
        None
    }

    #[inline(always)]
    fn cast_series(s: &Series) -> Cow<'_, Series> {
        Cow::Owned(s.cast(&DataType::Binary).unwrap())
    }

    #[inline(always)]
    fn combine(a: &mut Self::Value, b: &Self::Value) {
        Self::reduce_one(a, b.as_deref())
    }

    #[inline(always)]
    fn reduce_one(a: &mut Self::Value, b: Option<&[u8]>) {
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
    fn reduce_ca(v: &mut Self::Value, ca: &BinaryChunked) {
        Self::reduce_one(v, ca.max_binary())
    }

    #[inline(always)]
    fn finish(v: Vec<Self::Value>, m: Option<Bitmap>, dtype: &DataType) -> PolarsResult<Series> {
        assert!(m.is_none()); // This should only be used with VecGroupedReduction.
        let ca: BinaryChunked = v.into_iter().collect_ca(PlSmallStr::EMPTY);
        ca.into_series().cast(dtype)
    }
}

#[derive(Default)]
pub struct BoolMinGroupedReduction {
    values: MutableBitmap,
    mask: MutableBitmap,
}

impl GroupedReduction for BoolMinGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, true);
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &DataType::Boolean);
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if !ca.all() {
            self.values.set(group_idx as usize, false);
        }
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
        assert!(values.dtype() == &DataType::Boolean);
        assert!(values.len() == group_idxs.len());
        let ca: &BooleanChunked = values.as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, ov) in group_idxs.iter().zip(ca.iter()) {
                self.values
                    .and_pos_unchecked(*g as usize, ov.unwrap_or(true));
                self.mask.or_pos_unchecked(*g as usize, ov.is_some());
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
        assert!(self.values.len() == other.values.len());
        assert!(self.mask.len() == other.mask.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, (v, o)) in group_idxs
                .iter()
                .zip(other.values.iter().zip(other.mask.iter()))
            {
                self.values.and_pos_unchecked(*g as usize, v);
                self.mask.or_pos_unchecked(*g as usize, o);
            }
        }
        Ok(())
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
}

impl GroupedReduction for BoolMaxGroupedReduction {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.values.resize(num_groups as usize, false);
        self.mask.resize(num_groups as usize, false);
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &DataType::Boolean);
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if ca.any() {
            self.values.set(group_idx as usize, true);
        }
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
        assert!(values.dtype() == &DataType::Boolean);
        assert!(values.len() == group_idxs.len());
        let ca: &BooleanChunked = values.as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, ov) in group_idxs.iter().zip(ca.iter()) {
                self.values
                    .or_pos_unchecked(*g as usize, ov.unwrap_or(false));
                self.mask.or_pos_unchecked(*g as usize, ov.is_some());
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
        assert!(self.values.len() == other.values.len());
        assert!(self.mask.len() == other.mask.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, (v, o)) in group_idxs
                .iter()
                .zip(other.values.iter().zip(other.mask.iter()))
            {
                self.values.or_pos_unchecked(*g as usize, v);
                self.mask.or_pos_unchecked(*g as usize, o);
            }
        }
        Ok(())
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
