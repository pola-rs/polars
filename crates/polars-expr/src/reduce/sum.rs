use std::borrow::Cow;

use arrow::array::PrimitiveArray;
use num_traits::Zero;

use super::*;

pub struct SumReduce<T: PolarsNumericType> {
    sums: Vec<T::Native>,
    in_dtype: DataType,
}

impl<T: PolarsNumericType> SumReduce<T> {
    fn new(in_dtype: DataType) -> Self {
        SumReduce {
            sums: Vec::new(),
            in_dtype,
        }
    }
}

pub fn new_sum_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    match dtype {
        Boolean => Box::new(SumReduce::<IdxType>::new(dtype)),
        Int8 | UInt8 | Int16 | UInt16 => Box::new(SumReduce::<Int64Type>::new(dtype)),
        UInt32 => Box::new(SumReduce::<UInt32Type>::new(dtype)),
        UInt64 => Box::new(SumReduce::<UInt64Type>::new(dtype)),
        Int32 => Box::new(SumReduce::<Int32Type>::new(dtype)),
        Int64 => Box::new(SumReduce::<Int64Type>::new(dtype)),
        Float32 => Box::new(SumReduce::<Float32Type>::new(dtype)),
        Float64 => Box::new(SumReduce::<Float64Type>::new(dtype)),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(SumReduce::<Int128Type>::new(dtype)),
        Duration(_) => Box::new(SumReduce::<Int64Type>::new(dtype)),
        // For compatibility with the current engine, should probably be an error.
        String | Binary => Box::new(super::NullGroupedReduction::new(dtype)),
        _ => unimplemented!("{dtype:?} is not supported by sum reduction"),
    }
}

fn cast_sum_input<'a>(s: &'a Series, dt: &DataType) -> PolarsResult<Cow<'a, Series>> {
    use DataType::*;
    match dt {
        Boolean => Ok(Cow::Owned(s.cast(&IDX_DTYPE)?)),
        Int8 | UInt8 | Int16 | UInt16 => Ok(Cow::Owned(s.cast(&Int64)?)),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Ok(Cow::Owned(
            s.decimal().unwrap().physical().clone().into_series(),
        )),
        #[cfg(feature = "dtype-duration")]
        Duration(_) => Ok(Cow::Owned(
            s.duration().unwrap().physical().clone().into_series(),
        )),
        _ => Ok(Cow::Borrowed(s)),
    }
}

fn out_dtype(in_dtype: &DataType) -> DataType {
    use DataType::*;
    match in_dtype {
        Boolean => IDX_DTYPE,
        Int8 | UInt8 | Int16 | UInt16 => Int64,
        dt => dt.clone(),
    }
}

impl<T> GroupedReduction for SumReduce<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native> + IntoSeries,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            sums: Vec::new(),
            in_dtype: self.in_dtype.clone(),
        })
    }

    fn reserve(&mut self, additional: usize) {
        self.sums.reserve(additional);
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.sums.resize(num_groups as usize, T::Native::zero());
    }

    fn update_group(
        &mut self,
        values: &Series,
        group_idx: IdxSize,
        _seq_id: u64,
    ) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &self.in_dtype);
        let values = cast_sum_input(values, &self.in_dtype)?;
        let ca: &ChunkedArray<T> = values.as_ref().as_ref().as_ref();
        self.sums[group_idx as usize] += ChunkAgg::sum(ca).unwrap_or(T::Native::zero());
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
        _seq_id: u64,
    ) -> PolarsResult<()> {
        // TODO: we should really implement a sum-as-other-type operation instead
        // of doing this materialized cast.
        assert!(values.dtype() == &self.in_dtype);
        let values = cast_sum_input(values, &self.in_dtype)?;
        assert!(values.len() == group_idxs.len());
        let ca: &ChunkedArray<T> = values.as_ref().as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(ca.iter()) {
                *self.sums.get_unchecked_mut(*g as usize) += v.unwrap_or(T::Native::zero());
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
        assert!(other.sums.len() == group_idxs.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(other.sums.iter()) {
                *self.sums.get_unchecked_mut(*g as usize) += *v;
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
                *self.sums.get_unchecked_mut(*g as usize) += *other.sums.get_unchecked(*i as usize);
            }
        }
        Ok(())
    }

    unsafe fn partition(
        self: Box<Self>,
        partition_sizes: &[IdxSize],
        partition_idxs: &[IdxSize],
    ) -> Vec<Box<dyn GroupedReduction>> {
        partition::partition_vec(self.sums, partition_sizes, partition_idxs)
            .into_iter()
            .map(|sums| {
                Box::new(Self {
                    sums,
                    in_dtype: self.in_dtype.clone(),
                }) as _
            })
            .collect()
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let v = core::mem::take(&mut self.sums);
        let arr = Box::new(PrimitiveArray::<T::Native>::from_vec(v));
        Ok(unsafe {
            Series::from_chunks_and_dtype_unchecked(
                PlSmallStr::EMPTY,
                vec![arr],
                &out_dtype(&self.in_dtype),
            )
        })
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
