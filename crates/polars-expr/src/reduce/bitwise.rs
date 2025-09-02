use std::ops::{BitAnd, BitOr, BitXor, Not};

use arrow::array::BooleanArray;
use arrow::types::NativeType;
use num_traits::Zero;
use polars_compute::bitwise::BitwiseKernel;
use polars_core::with_match_physical_integer_polars_type;

use super::*;
use crate::reduce::min_max::{BoolMaxGroupedReduction, BoolMinGroupedReduction};

pub fn new_bitwise_and_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecMaskGroupedReduction as VMGR;
    match dtype {
        Boolean => Box::new(BoolMinGroupedReduction::default()),
        _ if dtype.is_integer() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<BitwiseAnd<$T>>::new()))
            })
        },
        _ => unimplemented!(),
    }
}

pub fn new_bitwise_or_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecMaskGroupedReduction as VMGR;
    match dtype {
        Boolean => Box::new(BoolMaxGroupedReduction::default()),
        _ if dtype.is_integer() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<BitwiseOr<$T>>::new()))
            })
        },
        _ => unimplemented!(),
    }
}

pub fn new_bitwise_xor_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    use VecMaskGroupedReduction as VMGR;
    match dtype {
        Boolean => Box::new(BoolXorGroupedReduction::default()),
        _ if dtype.is_integer() => {
            with_match_physical_integer_polars_type!(dtype.to_physical(), |$T| {
                Box::new(VMGR::new(dtype, NumReducer::<BitwiseXor<$T>>::new()))
            })
        },
        _ => unimplemented!(),
    }
}

struct BitwiseAnd<T>(PhantomData<T>);
struct BitwiseOr<T>(PhantomData<T>);
struct BitwiseXor<T>(PhantomData<T>);

impl<T> NumericReduction for BitwiseAnd<T>
where
    T: PolarsNumericType,
    T::Native: BitAnd<Output = T::Native> + Not<Output = T::Native> + Zero,
    T::Native: NativeType,
    PrimitiveArray<T::Native>: BitwiseKernel<Scalar = T::Native>,
{
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        !T::Native::zero() // all_ones(), the identity element
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        a & b
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<Self::Dtype>) -> Option<T::Native> {
        ca.and_reduce()
    }
}

impl<T> NumericReduction for BitwiseOr<T>
where
    T: PolarsNumericType,
    T::Native: BitOr<Output = T::Native> + Zero,
    T::Native: NativeType,
    PrimitiveArray<T::Native>: BitwiseKernel<Scalar = T::Native>,
{
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        T::Native::zero() // all_zeroes(), the identity element
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        a | b
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<Self::Dtype>) -> Option<T::Native> {
        ca.or_reduce()
    }
}

impl<T> NumericReduction for BitwiseXor<T>
where
    T: PolarsNumericType,
    T::Native: BitXor<Output = T::Native> + Zero,
    T::Native: NativeType,
    PrimitiveArray<T::Native>: BitwiseKernel<Scalar = T::Native>,
{
    type Dtype = T;

    #[inline(always)]
    fn init() -> T::Native {
        T::Native::zero() // all_zeroes(), the identity element
    }

    #[inline(always)]
    fn combine(a: T::Native, b: T::Native) -> T::Native {
        a ^ b
    }

    #[inline(always)]
    fn reduce_ca(ca: &ChunkedArray<Self::Dtype>) -> Option<T::Native> {
        ca.xor_reduce()
    }
}

#[derive(Default)]
struct BoolXorGroupedReduction {
    values: MutableBitmap,
    mask: MutableBitmap,
    evicted_values: BitmapBuilder,
    evicted_mask: BitmapBuilder,
}

impl GroupedReduction for BoolXorGroupedReduction {
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
        assert!(values.dtype() == &DataType::Boolean);
        let values = values.as_materialized_series_maintain_scalar();
        let ca: &BooleanChunked = values.as_ref().as_ref();
        if let Some(value) = ca.xor_reduce() {
            // SAFETY: indices are in-bounds guaranteed by trait
            unsafe {
                self.values.xor_pos_unchecked(group_idx as usize, value);
            }
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
                    self.values.xor_pos_unchecked(g.idx(), ov.unwrap_or(false));
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
                    .xor_pos_unchecked(*g as usize, other.values.get_unchecked(*i as usize));
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
