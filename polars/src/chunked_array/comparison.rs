use crate::prelude::*;
use arrow::array::{ArrayRef, BooleanArray, BooleanBuilder, PrimitiveArray, StringArray};
use arrow::compute;
use num::{Num, NumCast, ToPrimitive};
use std::sync::Arc;

/// Compare [Series](series/series/enum.Series.html)
///
/// and [ChunkedArray](series/chunked_array/struct.ChunkedArray.html)'s and get a `boolean` mask that
/// can be use to filter rows.
///
/// # Example
///
/// ```
/// use polars::prelude::*;
/// fn filter_all_ones(df: &DataFrame) -> Result<DataFrame> {
///     let mask = df
///     .column("column_a")
///     .ok_or(PolarsError::NotFound)?
///     .eq(1);
///
///     df.filter(&mask)
/// }
/// ```
pub trait CmpOps<Rhs> {
    /// Check for equality.
    fn eq(&self, rhs: Rhs) -> BooleanChunked;

    /// Check for inequality.
    fn neq(&self, rhs: Rhs) -> BooleanChunked;

    /// Greater than comparison.
    fn gt(&self, rhs: Rhs) -> BooleanChunked;

    /// Greater than or equal comparison.
    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked;

    /// Less than comparison.
    fn lt(&self, rhs: Rhs) -> BooleanChunked;

    /// Less than or equal comparison
    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked;
}

impl<T> ChunkedArray<T>
where
    T: PolarNumericType,
{
    /// First ensure that the chunks of lhs and rhs match and then iterates over the chunks and applies
    /// the comparison operator.
    fn comparison(
        &self,
        rhs: &ChunkedArray<T>,
        operator: impl Fn(&PrimitiveArray<T>, &PrimitiveArray<T>) -> arrow::error::Result<BooleanArray>,
    ) -> Result<BooleanChunked> {
        let chunks = self
            .downcast_chunks()
            .iter()
            .zip(rhs.downcast_chunks())
            .map(|(left, right)| {
                let arr_res = operator(left, right);
                let arr = match arr_res {
                    Ok(arr) => arr,
                    Err(e) => return Err(PolarsError::ArrowError(e)),
                };
                Ok(Arc::new(arr) as ArrayRef)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ChunkedArray::new_from_chunks("", chunks))
    }
}

impl<T> CmpOps<&ChunkedArray<T>> for ChunkedArray<T>
where
    T: PolarNumericType,
{
    fn eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            // should not fail if arrays are equal
            self.comparison(rhs, compute::eq)
                .expect("should not fail implementation error")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::neq).expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
        }
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt).expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt_eq)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt).expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt_eq)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

impl Utf8Chunked {
    fn comparison(
        &self,
        rhs: &Utf8Chunked,
        operator: impl Fn(&StringArray, &StringArray) -> arrow::error::Result<BooleanArray>,
    ) -> Result<BooleanChunked> {
        let chunks = self
            .chunks
            .iter()
            .zip(&rhs.chunks)
            .map(|(left, right)| {
                let left = left
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("could not downcast one of the chunks");
                let right = right
                    .as_any()
                    .downcast_ref::<StringArray>()
                    .expect("could not downcast one of the chunks");
                let arr_res = operator(left, right);
                let arr = match arr_res {
                    Ok(arr) => arr,
                    Err(e) => return Err(PolarsError::ArrowError(e)),
                };
                Ok(Arc::new(arr) as ArrayRef)
            })
            .collect::<Result<Vec<_>>>()?;

        Ok(ChunkedArray::new_from_chunks("", chunks))
    }
}

impl CmpOps<&Utf8Chunked> for Utf8Chunked {
    fn eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::eq_utf8)
                .expect("should not fail")
        } else {
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left, right)| left == right)
                .collect()
        }
    }

    fn neq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::neq_utf8)
                .expect("should not fail")
        } else {
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left, right)| left != right)
                .collect()
        }
    }

    fn gt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt_utf8)
                .expect("should not fail")
        } else {
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left, right)| left > right)
                .collect()
        }
    }

    fn gt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt_eq_utf8)
                .expect("should not fail")
        } else {
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left, right)| left >= right)
                .collect()
        }
    }

    fn lt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt_utf8)
                .expect("should not fail")
        } else {
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left, right)| left < right)
                .collect()
        }
    }

    fn lt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt_eq_utf8)
                .expect("should not fail")
        } else {
            self.into_iter()
                .zip(rhs.into_iter())
                .map(|(left, right)| left <= right)
                .collect()
        }
    }
}

fn cmp_chunked_array_to_num<T, Rhs>(
    ca: &ChunkedArray<T>,
    cmp_fn: &dyn Fn(Rhs) -> bool,
) -> Result<BooleanChunked>
where
    T: PolarNumericType,
    T::Native: ToPrimitive,
    Rhs: Num + NumCast,
{
    // TODO: Doesnt to do null checks
    let chunks = ca
        .downcast_chunks()
        .iter()
        .map(|&a| {
            let mut builder = BooleanBuilder::new(a.len());

            for i in 0..a.len() {
                let val = a.value(i);
                let val = Rhs::from(val);
                let val = match val {
                    Some(val) => val,
                    None => return Err(PolarsError::DataTypeMisMatch),
                };
                builder.append_value(cmp_fn(val))?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        })
        .collect::<Result<Vec<_>>>()?;

    Ok(BooleanChunked::new_from_chunks("", chunks))
}

pub trait NumComp: Num + NumCast + PartialOrd {}

impl NumComp for f32 {}
impl NumComp for f64 {}
impl NumComp for i8 {}
impl NumComp for i16 {}
impl NumComp for i32 {}
impl NumComp for i64 {}
impl NumComp for u8 {}
impl NumComp for u16 {}
impl NumComp for u32 {}
impl NumComp for u64 {}

impl<T, Rhs> CmpOps<Rhs> for ChunkedArray<T>
where
    T: PolarNumericType,
    T::Native: ToPrimitive,
    Rhs: NumComp,
{
    fn eq(&self, rhs: Rhs) -> BooleanChunked {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs == rhs).expect("should not fail")
    }

    fn neq(&self, rhs: Rhs) -> BooleanChunked {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs != rhs).expect("should not fail")
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs > rhs).expect("should not fail")
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs >= rhs).expect("should not fail")
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs < rhs).expect("should not fail")
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs <= rhs).expect("should not fail")
    }
}

fn cmp_utf8chunked_to_str(ca: &Utf8Chunked, cmp_fn: &dyn Fn(&str) -> bool) -> BooleanChunked {
    ca.into_iter().map(cmp_fn).collect()
}

impl CmpOps<&str> for Utf8Chunked {
    fn eq(&self, rhs: &str) -> BooleanChunked {
        cmp_utf8chunked_to_str(self, &|lhs| lhs == rhs)
    }

    fn neq(&self, rhs: &str) -> BooleanChunked {
        cmp_utf8chunked_to_str(self, &|lhs| lhs != rhs)
    }

    fn gt(&self, rhs: &str) -> BooleanChunked {
        cmp_utf8chunked_to_str(self, &|lhs| lhs > rhs)
    }

    fn gt_eq(&self, rhs: &str) -> BooleanChunked {
        cmp_utf8chunked_to_str(self, &|lhs| lhs >= rhs)
    }

    fn lt(&self, rhs: &str) -> BooleanChunked {
        cmp_utf8chunked_to_str(self, &|lhs| lhs < rhs)
    }

    fn lt_eq(&self, rhs: &str) -> BooleanChunked {
        cmp_utf8chunked_to_str(self, &|lhs| lhs <= rhs)
    }
}
