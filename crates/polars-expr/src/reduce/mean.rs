use std::marker::PhantomData;

use num_traits::{AsPrimitive, Zero};

use super::*;

pub fn new_mean_reduction(dtype: DataType) -> Box<dyn GroupedReduction> {
    use DataType::*;
    match dtype {
        Boolean => Box::new(BoolMeanReduce::default()),
        UInt8 => Box::new(NumMeanReduce::<UInt8Type>::new(dtype)),
        UInt16 => Box::new(NumMeanReduce::<UInt16Type>::new(dtype)),
        UInt32 => Box::new(NumMeanReduce::<UInt32Type>::new(dtype)),
        UInt64 => Box::new(NumMeanReduce::<UInt64Type>::new(dtype)),
        Int8 => Box::new(NumMeanReduce::<Int8Type>::new(dtype)),
        Int16 => Box::new(NumMeanReduce::<Int16Type>::new(dtype)),
        Int32 => Box::new(NumMeanReduce::<Int32Type>::new(dtype)),
        Int64 => Box::new(NumMeanReduce::<Int64Type>::new(dtype)),
        Float32 => Box::new(NumMeanReduce::<Float32Type>::new(dtype)),
        Float64 => Box::new(NumMeanReduce::<Float64Type>::new(dtype)),
        #[cfg(feature = "dtype-decimal")]
        Decimal(_, _) => Box::new(NumMeanReduce::<Int128Type>::new(dtype)),
        Duration(_) => Box::new(NumMeanReduce::<Int64Type>::new(dtype)),
        Datetime(_, _) => Box::new(NumMeanReduce::<Int64Type>::new(dtype)),
        Date => Box::new(NumMeanReduce::<Int32Type>::new(dtype)),
        Time => Box::new(NumMeanReduce::<Int64Type>::new(dtype)),
        _ => unimplemented!(),
    }
}

pub struct NumMeanReduce<T> {
    groups: Vec<(f64, usize)>,
    in_dtype: DataType,
    phantom: PhantomData<T>,
}

impl<T> NumMeanReduce<T> {
    fn new(in_dtype: DataType) -> Self {
        NumMeanReduce {
            groups: Vec::new(),
            in_dtype,
            phantom: PhantomData,
        }
    }
}

fn finish_output(values: Vec<(f64, usize)>, dtype: &DataType) -> Series {
    match dtype {
        DataType::Float32 => {
            let ca: Float32Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64) as f32))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        dt if dt.is_numeric() => {
            let ca: Float64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| s / c as f64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        DataType::Decimal(_prec, scale) => {
            let inv_scale_factor = 1.0 / 10u128.pow(scale.unwrap() as u32) as f64;
            let ca: Float64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| s / c as f64 * inv_scale_factor))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series()
        },
        DataType::Date => {
            const MS_IN_DAY: i64 = 86_400_000;
            let ca: Int64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64 * MS_IN_DAY as f64) as i64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_datetime(TimeUnit::Milliseconds, None).into_series()
        },
        DataType::Datetime(_, _) | DataType::Duration(_) | DataType::Time => {
            let ca: Int64Chunked = values
                .into_iter()
                .map(|(s, c)| (c != 0).then(|| (s / c as f64) as i64))
                .collect_ca(PlSmallStr::EMPTY);
            ca.into_series().cast(dtype).unwrap()
        },
        _ => unimplemented!(),
    }
}

impl<T> GroupedReduction for NumMeanReduce<T>
where
    T: PolarsNumericType,
    ChunkedArray<T>: ChunkAgg<T::Native> + IntoSeries,
{
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self {
            groups: Vec::new(),
            in_dtype: self.in_dtype.clone(),
            phantom: PhantomData,
        })
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.groups.resize(num_groups as usize, (0.0, 0));
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<T> = values.as_ref().as_ref().as_ref();
        let grp = &mut self.groups[group_idx as usize];
        grp.0 += ChunkAgg::_sum_as_f64(ca);
        grp.1 += ca.len() - ca.null_count();
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        assert!(values.dtype() == &self.in_dtype);
        assert!(values.len() == group_idxs.len());
        let values = values.to_physical_repr();
        let ca: &ChunkedArray<T> = values.as_ref().as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(ca.iter()) {
                let grp = self.groups.get_unchecked_mut(*g as usize);
                grp.0 += v.unwrap_or(T::Native::zero()).as_();
                grp.1 += v.is_some() as usize;
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
        assert!(self.groups.len() == other.groups.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(other.groups.iter()) {
                let grp = self.groups.get_unchecked_mut(*g as usize);
                grp.0 += v.0;
                grp.1 += v.1;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        Ok(finish_output(
            core::mem::take(&mut self.groups),
            &self.in_dtype,
        ))
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}


#[derive(Default)]
pub struct BoolMeanReduce {
    groups: Vec<(usize, usize)>,
}

impl GroupedReduction for BoolMeanReduce {
    fn new_empty(&self) -> Box<dyn GroupedReduction> {
        Box::new(Self::default())
    }

    fn resize(&mut self, num_groups: IdxSize) {
        self.groups.resize(num_groups as usize, (0, 0));
    }

    fn update_group(&mut self, values: &Series, group_idx: IdxSize) -> PolarsResult<()> {
        let ca: &BooleanChunked = values.as_ref().as_ref();
        let grp = &mut self.groups[group_idx as usize];
        grp.0 += ca.sum().unwrap_or(0) as usize;
        grp.1 += ca.len() - ca.null_count();
        Ok(())
    }

    unsafe fn update_groups(
        &mut self,
        values: &Series,
        group_idxs: &[IdxSize],
    ) -> PolarsResult<()> {
        assert!(values.len() == group_idxs.len());
        let ca: &BooleanChunked = values.as_ref().as_ref();
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(ca.iter()) {
                let grp = self.groups.get_unchecked_mut(*g as usize);
                grp.0 += v.unwrap_or(false) as usize;
                grp.1 += v.is_some() as usize;
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
        assert!(self.groups.len() == other.groups.len());
        unsafe {
            // SAFETY: indices are in-bounds guaranteed by trait.
            for (g, v) in group_idxs.iter().zip(other.groups.iter()) {
                let grp = self.groups.get_unchecked_mut(*g as usize);
                grp.0 += v.0;
                grp.1 += v.1;
            }
        }
        Ok(())
    }

    fn finalize(&mut self) -> PolarsResult<Series> {
        let ca: Float64Chunked = self.groups.drain(..).map(|(s, c)| s as f64 / c as f64).collect_ca(PlSmallStr::EMPTY);
        Ok(ca.into_series())
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}