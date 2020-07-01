use crate::prelude::*;
use arrow::array::{Array, ArrayRef, BooleanArray, BooleanBuilder, PrimitiveArray, StringArray};
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
///     .eq(1)?;
///
///     df.filter(&mask)
/// }
/// ```
pub trait CmpOps<Rhs> {
    /// Check for equality.
    fn eq(&self, rhs: Rhs) -> Result<BooleanChunked>;

    /// Check for inequality.
    fn neq(&self, rhs: Rhs) -> Result<BooleanChunked>;

    /// Greater than comparison.
    fn gt(&self, rhs: Rhs) -> Result<BooleanChunked>;

    /// Greater than or equal comparison.
    fn gt_eq(&self, rhs: Rhs) -> Result<BooleanChunked>;

    /// Less than comparison.
    fn lt(&self, rhs: Rhs) -> Result<BooleanChunked>;

    /// Less than or equal comparison
    fn lt_eq(&self, rhs: Rhs) -> Result<BooleanChunked>;
}

/// Forced comparisons. Results are unwrapped.
pub trait ForceCmpOps<Rhs>: CmpOps<Rhs> {
    fn f_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.eq(rhs).expect("could not cmp")
    }
    fn f_neq(&self, rhs: Rhs) -> BooleanChunked {
        self.neq(rhs).expect("could not cmp")
    }
    fn f_gt(&self, rhs: Rhs) -> BooleanChunked {
        self.gt(rhs).expect("could not cmp")
    }
    fn f_gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.gt_eq(rhs).expect("could not cmp")
    }
    fn f_lt(&self, rhs: Rhs) -> BooleanChunked {
        self.lt(rhs).expect("could not cmp")
    }
    fn f_lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        self.lt_eq(rhs).expect("could not cmp")
    }
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
        let opt = self.optional_rechunk(rhs)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };

        let chunks = left
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
    fn eq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::eq)
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::neq)
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::gt)
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::gt_eq)
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::lt)
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::lt_eq)
    }
}

impl<T: PolarNumericType> ForceCmpOps<&ChunkedArray<T>> for ChunkedArray<T> {}

/// Auxiliary trait for CmpOps trait
pub trait BoundedToUtf8 {}
impl BoundedToUtf8 for datatypes::Utf8Type {}

impl Utf8Chunked {
    fn comparison<T: BoundedToUtf8>(
        &self,
        rhs: &ChunkedArray<T>,
        operator: impl Fn(&StringArray, &StringArray) -> arrow::error::Result<BooleanArray>,
    ) -> Result<BooleanChunked> {
        let opt = self.optional_rechunk(rhs)?;
        let left = match &opt {
            Some(a) => a,
            None => self,
        };

        let chunks = left
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

impl<T> CmpOps<&ChunkedArray<T>> for Utf8Chunked
where
    T: BoundedToUtf8,
{
    fn eq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::eq_utf8)
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::neq_utf8)
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::gt_utf8)
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::gt_eq_utf8)
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::lt_utf8)
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> Result<BooleanChunked> {
        self.comparison(rhs, compute::lt_eq_utf8)
    }
}

impl<T: BoundedToUtf8> ForceCmpOps<&ChunkedArray<T>> for Utf8Chunked {}

fn cmp_chunked_array_to_num<T, Rhs>(
    ca: &ChunkedArray<T>,
    cmp_fn: &dyn Fn(Rhs) -> bool,
) -> Result<BooleanChunked>
where
    T: PolarNumericType,
    T::Native: ToPrimitive,
    Rhs: Num + NumCast,
{
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
    fn eq(&self, rhs: Rhs) -> Result<BooleanChunked> {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs == rhs)
    }

    fn neq(&self, rhs: Rhs) -> Result<BooleanChunked> {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs != rhs)
    }

    fn gt(&self, rhs: Rhs) -> Result<BooleanChunked> {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs > rhs)
    }

    fn gt_eq(&self, rhs: Rhs) -> Result<BooleanChunked> {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs >= rhs)
    }

    fn lt(&self, rhs: Rhs) -> Result<BooleanChunked> {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs < rhs)
    }

    fn lt_eq(&self, rhs: Rhs) -> Result<BooleanChunked> {
        cmp_chunked_array_to_num(self, &|lhs: Rhs| lhs <= rhs)
    }
}

impl<T, Rhs> ForceCmpOps<Rhs> for ChunkedArray<T>
where
    T: PolarNumericType,
    T::Native: ToPrimitive,
    Rhs: NumComp,
{
}

fn cmp_chunked_array_to_str(
    ca: &Utf8Chunked,
    cmp_fn: &dyn Fn(&str) -> bool,
) -> Result<BooleanChunked> {
    let chunks = ca
        .downcast_chunks()
        .iter()
        .map(|a| {
            let mut builder = BooleanBuilder::new(a.len());

            for i in 0..a.len() {
                let val = a.value(i);
                builder.append_value(cmp_fn(val))?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanChunked::new_from_chunks("", chunks))
}

impl CmpOps<&str> for Utf8Chunked {
    fn eq(&self, rhs: &str) -> Result<BooleanChunked> {
        cmp_chunked_array_to_str(self, &|lhs| lhs == rhs)
    }

    fn neq(&self, rhs: &str) -> Result<BooleanChunked> {
        cmp_chunked_array_to_str(self, &|lhs| lhs != rhs)
    }

    fn gt(&self, rhs: &str) -> Result<BooleanChunked> {
        cmp_chunked_array_to_str(self, &|lhs| lhs > rhs)
    }

    fn gt_eq(&self, rhs: &str) -> Result<BooleanChunked> {
        cmp_chunked_array_to_str(self, &|lhs| lhs >= rhs)
    }

    fn lt(&self, rhs: &str) -> Result<BooleanChunked> {
        cmp_chunked_array_to_str(self, &|lhs| lhs < rhs)
    }

    fn lt_eq(&self, rhs: &str) -> Result<BooleanChunked> {
        cmp_chunked_array_to_str(self, &|lhs| lhs <= rhs)
    }
}

impl ForceCmpOps<&str> for Utf8Chunked {}

fn cmp_chunked_array_to_boolarr(
    ca: &BooleanChunked,
    cmp_fn: &dyn Fn(bool, bool) -> bool,
    rhs: &[bool],
) -> Result<BooleanChunked> {
    let chunks = ca
        .downcast_chunks()
        .iter()
        .map(|a| {
            let mut builder = BooleanBuilder::new(a.len());

            for i in 0..a.len() {
                let val = a.value(i);
                builder.append_value(cmp_fn(val, rhs[i]))?;
            }
            Ok(Arc::new(builder.finish()) as ArrayRef)
        })
        .collect::<Result<Vec<_>>>()?;
    Ok(BooleanChunked::new_from_chunks("", chunks))
}

impl CmpOps<&[bool]> for BooleanChunked {
    fn eq(&self, rhs: &[bool]) -> Result<BooleanChunked> {
        cmp_chunked_array_to_boolarr(self, &|lhs, rhs_i| lhs == rhs_i, rhs)
    }

    fn neq(&self, rhs: &[bool]) -> Result<BooleanChunked> {
        cmp_chunked_array_to_boolarr(self, &|lhs, rhs_i| lhs != rhs_i, rhs)
    }

    fn gt(&self, rhs: &[bool]) -> Result<BooleanChunked> {
        cmp_chunked_array_to_boolarr(self, &|lhs, rhs_i| lhs > rhs_i, rhs)
    }

    fn gt_eq(&self, rhs: &[bool]) -> Result<BooleanChunked> {
        cmp_chunked_array_to_boolarr(self, &|lhs, rhs_i| lhs >= rhs_i, rhs)
    }

    fn lt(&self, rhs: &[bool]) -> Result<BooleanChunked> {
        cmp_chunked_array_to_boolarr(self, &|lhs, rhs_i| lhs < rhs_i, rhs)
    }

    fn lt_eq(&self, rhs: &[bool]) -> Result<BooleanChunked> {
        cmp_chunked_array_to_boolarr(self, &|lhs, rhs_i| lhs <= rhs_i, rhs)
    }
}

impl ForceCmpOps<&[bool]> for BooleanChunked {}

#[cfg(test)]
mod test {
    use crate::prelude::*;

    #[test]
    fn utf8_cmp() {
        let a = ChunkedArray::<datatypes::Utf8Type>::new_utf8_from_slice("a", &["hello", "world"]);
        let b = ChunkedArray::<datatypes::Utf8Type>::new_utf8_from_slice("a", &["hello", "world"]);
        let sum_true = a.eq(&b).unwrap().into_iter().fold(0, |acc, opt| match opt {
            Some(b) => acc + b as i32,
            None => acc,
        });

        assert_eq!(2, sum_true)
    }
}
