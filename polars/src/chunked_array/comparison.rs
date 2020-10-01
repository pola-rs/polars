use crate::{prelude::*, utils::Xob};
use arrow::{
    array::{ArrayRef, BooleanArray, PrimitiveArray, StringArray},
    compute,
};
use num::{Num, NumCast, ToPrimitive};
use std::ops::{BitAnd, BitOr, Not};
use std::sync::Arc;

impl<T> ChunkedArray<T>
where
    T: PolarsNumericType,
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

macro_rules! impl_eq_missing {
    ($self:ident, $rhs:ident) => {{
        match ($self.null_count(), $rhs.null_count()) {
            (0, 0) => $self
                .into_no_null_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(opt_a, opt_b)| opt_a == opt_b)
                .collect(),
            (_, _) => $self
                .into_iter()
                .zip($rhs)
                .map(|(opt_a, opt_b)| opt_a == opt_b)
                .collect(),
        }
    }};
}

impl<T> ChunkCompare<&ChunkedArray<T>> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast + NumComp + ToPrimitive,
{
    fn eq_missing(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            // should not fail if arrays are equal
            self.comparison(rhs, compute::eq).expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.neq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::neq)
                .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
        }
    }

    fn gt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt).expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt_eq)
                .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt).expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &ChunkedArray<T>) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt_eq)
                .expect("should not fail.")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

impl ChunkCompare<&BooleanChunked> for BooleanChunked {
    fn eq_missing(&self, rhs: &BooleanChunked) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.clone(),
                    false => self.not(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.not(),
                    false => self.clone(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
        }
    }

    fn gt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => BooleanChunked::full("", false, self.len()),
                    false => self.clone(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.clone(),
                    false => BooleanChunked::full("", true, self.len()),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => self.not(),
                    false => BooleanChunked::full("", false, self.len()),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &BooleanChunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                match value {
                    true => BooleanChunked::full("", true, self.len()),
                    false => self.not(),
                }
            } else {
                BooleanChunked::full("", false, self.len())
            }
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

impl ChunkCompare<&Utf8Chunked> for Utf8Chunked {
    fn eq_missing(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        impl_eq_missing!(self, rhs)
    }

    fn eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::eq_utf8)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, ==)
        }
    }

    fn neq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.neq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::neq_utf8)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, !=)
        }
    }

    fn gt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt_utf8)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >)
        }
    }

    fn gt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.gt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::gt_eq_utf8)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, >=)
        }
    }

    fn lt(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt_utf8)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <)
        }
    }

    fn lt_eq(&self, rhs: &Utf8Chunked) -> BooleanChunked {
        // broadcast
        if rhs.len() == 1 {
            if let Some(value) = rhs.get(0) {
                self.lt_eq(value)
            } else {
                BooleanChunked::full("", false, self.len())
            }
        }
        // same length
        else if self.chunk_id == rhs.chunk_id {
            self.comparison(rhs, compute::lt_eq_utf8)
                .expect("should not fail")
        } else {
            apply_operand_on_chunkedarray_by_iter!(self, rhs, <=)
        }
    }
}

fn cmp_chunked_array_to_num<T>(
    ca: &ChunkedArray<T>,
    cmp_fn: &dyn Fn(Option<T::Native>) -> bool,
) -> BooleanChunked
where
    T: PolarsNumericType,
{
    ca.into_iter().map(cmp_fn).collect()
}

fn cmp_chunked_array_to_num_no_null<T>(
    ca: &ChunkedArray<T>,
    cmp_fn: &dyn Fn(T::Native) -> bool,
) -> BooleanChunked
where
    T: PolarsNumericType,
{
    ca.into_no_null_iter().map(cmp_fn).collect()
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

impl<T, Rhs> ChunkCompare<Rhs> for ChunkedArray<T>
where
    T: PolarsNumericType,
    T::Native: NumCast,
    Rhs: NumComp + ToPrimitive,
{
    fn eq_missing(&self, rhs: Rhs) -> BooleanChunked {
        self.eq(rhs)
    }

    fn eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs = NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        match self.null_count() {
            0 => cmp_chunked_array_to_num_no_null(self, &|lhs| lhs == rhs),
            _ => cmp_chunked_array_to_num(self, &|lhs: Option<T::Native>| lhs == Some(rhs)),
        }
    }

    fn neq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs = NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        match self.null_count() {
            0 => cmp_chunked_array_to_num_no_null(self, &|lhs| lhs != rhs),
            _ => cmp_chunked_array_to_num(self, &|lhs: Option<T::Native>| lhs != Some(rhs)),
        }
    }

    fn gt(&self, rhs: Rhs) -> BooleanChunked {
        let rhs = NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        match self.null_count() {
            0 => cmp_chunked_array_to_num_no_null(self, &|lhs| lhs > rhs),
            _ => cmp_chunked_array_to_num(self, &|lhs: Option<T::Native>| lhs > Some(rhs)),
        }
    }

    fn gt_eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs = NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        match self.null_count() {
            0 => cmp_chunked_array_to_num_no_null(self, &|lhs| lhs >= rhs),
            _ => cmp_chunked_array_to_num(self, &|lhs: Option<T::Native>| lhs >= Some(rhs)),
        }
    }

    fn lt(&self, rhs: Rhs) -> BooleanChunked {
        let rhs = NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        match self.null_count() {
            0 => cmp_chunked_array_to_num_no_null(self, &|lhs| lhs < rhs),
            _ => cmp_chunked_array_to_num(self, &|lhs: Option<T::Native>| lhs < Some(rhs)),
        }
    }

    fn lt_eq(&self, rhs: Rhs) -> BooleanChunked {
        let rhs = NumCast::from(rhs).expect("could not cast to underlying chunkedarray type");
        match self.null_count() {
            0 => cmp_chunked_array_to_num_no_null(self, &|lhs| lhs <= rhs),
            _ => cmp_chunked_array_to_num(self, &|lhs: Option<T::Native>| lhs <= Some(rhs)),
        }
    }
}

fn cmp_utf8chunked_to_str(ca: &Utf8Chunked, cmp_fn: &dyn Fn(&str) -> bool) -> BooleanChunked {
    ca.into_iter()
        .map(|opt_s| match opt_s {
            None => false,
            Some(s) => cmp_fn(s),
        })
        .collect()
}

impl ChunkCompare<&str> for Utf8Chunked {
    fn eq_missing(&self, rhs: &str) -> BooleanChunked {
        self.eq(rhs)
    }

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

macro_rules! impl_cmp_largelist {
    ($self:ident, $rhs:ident, $cmp_method:ident) => {{
        match ($self.null_count(), $rhs.null_count()) {
            (0, 0) => {
                $self
                    .into_no_null_iter()
                    .zip($rhs.into_no_null_iter())
                    // TODO: use Xob to get rid of redundant Some<T>
                    .map(|(left, right)| Some(left.$cmp_method(&right)))
                    .collect()
            }
            (0, _) => $self
                .into_no_null_iter()
                .zip($rhs.into_iter())
                .map(|(left, opt_right)| opt_right.map(|right| left.$cmp_method(&right)))
                .collect(),
            (_, 0) => $self
                .into_iter()
                .zip($rhs.into_no_null_iter())
                .map(|(opt_left, right)| opt_left.map(|left| left.$cmp_method(&right)))
                .collect(),
            (_, _) => $self
                .into_iter()
                .zip($rhs.into_iter())
                .map(|(opt_left, opt_right)| match (opt_left, opt_right) {
                    (None, None) => None,
                    (None, Some(_)) => None,
                    (Some(_), None) => None,
                    (Some(left), Some(right)) => Some(left.$cmp_method(&right)),
                })
                .collect(),
        }
    }};
}

impl ChunkCompare<&LargeListChunked> for LargeListChunked {
    fn eq_missing(&self, rhs: &LargeListChunked) -> BooleanChunked {
        impl_cmp_largelist!(self, rhs, series_equal_missing)
    }

    fn eq(&self, rhs: &LargeListChunked) -> BooleanChunked {
        impl_cmp_largelist!(self, rhs, series_equal)
    }

    fn neq(&self, rhs: &LargeListChunked) -> BooleanChunked {
        self.eq(rhs).not()
    }

    // following are not implemented because gt, lt comparison of series don't make sense
    fn gt(&self, _rhs: &LargeListChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn gt_eq(&self, _rhs: &LargeListChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt(&self, _rhs: &LargeListChunked) -> BooleanChunked {
        unimplemented!()
    }

    fn lt_eq(&self, _rhs: &LargeListChunked) -> BooleanChunked {
        unimplemented!()
    }
}

impl BooleanChunked {
    /// First ensure that the chunks of lhs and rhs match and then iterates over the chunks and applies
    /// the comparison operator.
    fn bit_operation(
        &self,
        rhs: &BooleanChunked,
        operator: impl Fn(&BooleanArray, &BooleanArray) -> arrow::error::Result<BooleanArray>,
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

macro_rules! impl_bitwise_op  {
    ($self:ident, $rhs:ident, $arrow_method:ident, $op:tt) => {{
        if $self.chunk_id == $rhs.chunk_id {
            let result = $self.bit_operation($rhs, compute::$arrow_method);
            result.unwrap()
        } else {
            let ca = $self
                .into_iter()
                .zip($rhs.into_iter())
                .map(|(opt_left, opt_right)| match (opt_left, opt_right) {
                    (Some(left), Some(right)) => Some(left $op right),
                    _ => None,
                })
                .collect();
            ca
        }
    }}

}

impl BitOr for &BooleanChunked {
    type Output = BooleanChunked;

    fn bitor(self, rhs: Self) -> Self::Output {
        impl_bitwise_op!(self, rhs, or, |)
    }
}

impl BitOr for BooleanChunked {
    type Output = BooleanChunked;

    fn bitor(self, rhs: Self) -> Self::Output {
        (&self).bitor(&rhs)
    }
}

impl BitAnd for &BooleanChunked {
    type Output = BooleanChunked;

    fn bitand(self, rhs: Self) -> Self::Output {
        impl_bitwise_op!(self, rhs, and, &)
    }
}

impl BitAnd for BooleanChunked {
    type Output = BooleanChunked;

    fn bitand(self, rhs: Self) -> Self::Output {
        (&self).bitand(&rhs)
    }
}

impl Not for &BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        let chunks = self
            .downcast_chunks()
            .iter()
            .map(|a| {
                let arr = compute::not(a).expect("should not fail");
                Arc::new(arr) as ArrayRef
            })
            .collect::<Vec<_>>();
        ChunkedArray::new_from_chunks(self.name(), chunks)
    }
}

impl Not for BooleanChunked {
    type Output = BooleanChunked;

    fn not(self) -> Self::Output {
        (&self).not()
    }
}

pub trait CompToSeries {
    fn lt_series(&self, _rhs: &Series) -> BooleanChunked {
        unimplemented!()
    }

    fn gt_series(&self, _rhs: &Series) -> BooleanChunked {
        unimplemented!()
    }

    fn gt_eq_series(&self, _rhs: &Series) -> BooleanChunked {
        unimplemented!()
    }

    fn lt_eq_series(&self, _rhs: &Series) -> BooleanChunked {
        unimplemented!()
    }

    fn eq_series(&self, _rhs: &Series) -> BooleanChunked {
        unimplemented!()
    }

    fn neq_series(&self, _rhs: &Series) -> BooleanChunked {
        unimplemented!()
    }
}

macro_rules! impl_comp_to_series {
    ($SELF:ident, $METHOD_NAME:ident, $OPERATION:ident, $RHS:ident, $NAME:expr, $TYPE:ty) => {{
        match $SELF.unpack_series_matching_type($RHS) {
            Ok(ca) => $SELF.lt(ca),
            Err(_) => match $RHS.cast::<$TYPE>() {
                Ok(s) => $SELF.$METHOD_NAME(&s),
                Err(_) => BooleanChunked::full($NAME, false, $SELF.len()),
            },
        }
    }};
}

impl<T> CompToSeries for ChunkedArray<T>
where
    T: PolarsNumericType,
    // extra trait bound is required to call match the trait bounds of
    // impl<T> ChunkCompare<&ChunkedArray<T>> for ChunkedArray<T>
    T::Native: NumCast + NumComp + ToPrimitive,
{
    fn lt_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, lt_series, lt, rhs, "lt", T)
    }

    fn gt_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, gt_series, gt, rhs, "gt", T)
    }

    fn gt_eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, gt_eq_series, gt_eq, rhs, "gt_eq", T)
    }

    fn lt_eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, lt_eq_series, lt_eq, rhs, "lt_eq", T)
    }

    fn eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, eq_series, eq, rhs, "eq", T)
    }

    fn neq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, neq_series, neq, rhs, "neq", T)
    }
}

impl CompToSeries for BooleanChunked {
    fn lt_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, lt_series, lt, rhs, "lt", BooleanType)
    }

    fn gt_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, gt_series, gt, rhs, "gt", BooleanType)
    }

    fn gt_eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, gt_eq_series, gt_eq, rhs, "gt_eq", BooleanType)
    }

    fn lt_eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, lt_eq_series, lt_eq, rhs, "lt_eq", BooleanType)
    }

    fn eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, eq_series, eq, rhs, "eq", BooleanType)
    }

    fn neq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, neq_series, neq, rhs, "neq", BooleanType)
    }
}
impl CompToSeries for Utf8Chunked {
    fn lt_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, lt_series, lt, rhs, "lt", Utf8Type)
    }

    fn gt_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, gt_series, gt, rhs, "gt", Utf8Type)
    }
    fn gt_eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, gt_eq_series, gt_eq, rhs, "gt_eq", Utf8Type)
    }

    fn lt_eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, lt_eq_series, lt_eq, rhs, "lt_eq", Utf8Type)
    }

    fn eq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, eq_series, eq, rhs, "eq", Utf8Type)
    }

    fn neq_series(&self, rhs: &Series) -> BooleanChunked {
        impl_comp_to_series!(self, neq_series, neq, rhs, "neq", Utf8Type)
    }
}
impl CompToSeries for LargeListChunked {}

impl BooleanChunked {
    pub fn all_true(&self) -> bool {
        match self.sum() {
            None => false,
            Some(n) => (n as usize) == self.len(),
        }
    }
}

impl BooleanChunked {
    pub fn all_false(&self) -> bool {
        match self.sum() {
            None => false,
            Some(n) => (n as usize) == 0,
        }
    }
}

#[cfg(test)]
mod test {
    use super::super::{arithmetic::test::create_two_chunked, test::get_chunked_array};
    use crate::prelude::*;
    use itertools::Itertools;
    use std::iter::repeat;

    #[test]
    fn test_bitwise_ops() {
        let a = BooleanChunked::new_from_slice("a", &[true, false, false]);
        let b = BooleanChunked::new_from_opt_slice("b", &[Some(true), Some(true), None]);
        assert_eq!(
            Vec::from((&a | &b).unwrap()),
            &[Some(true), Some(true), None]
        );
        assert_eq!(
            Vec::from((&a & &b).unwrap()),
            &[Some(true), Some(false), None]
        );
        assert_eq!(Vec::from(!b), &[Some(false), Some(false), None]);
    }

    #[test]
    fn test_compare_chunk_diff() {
        let (a1, a2) = create_two_chunked();

        assert_eq!(
            a1.eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.neq(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.gt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.gt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a2.lt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(6).collect_vec()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
        assert_eq!(
            a2.lt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(6).collect_vec()
        );
    }

    #[test]
    fn test_equal_chunks() {
        let a1 = get_chunked_array();
        let a2 = get_chunked_array();

        assert_eq!(
            a1.eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.neq(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.gt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.gt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a2.lt_eq(&a1).into_iter().collect_vec(),
            repeat(Some(true)).take(3).collect_vec()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
        assert_eq!(
            a2.lt(&a1).into_iter().collect_vec(),
            repeat(Some(false)).take(3).collect_vec()
        );
    }

    #[test]
    fn test_null_handling() {
        // assert we comply with arrows way of handling null data
        // we check comparison on two arrays with one chunk and verify it is equal to a differently
        // chunked array comparison.

        // two same chunked arrays
        let a1: Int32Chunked = (&[Some(1), None, Some(3)]).iter().copied().collect();
        let a2: Int32Chunked = (&[Some(1), Some(2), Some(3)]).iter().copied().collect();

        let mut a2_2chunks: Int32Chunked = (&[Some(1), Some(2)]).iter().copied().collect();
        a2_2chunks.append(&(&[Some(3)]).iter().copied().collect());

        assert_eq!(
            a1.eq(&a2).into_iter().collect_vec(),
            a1.eq(&a2_2chunks).into_iter().collect_vec()
        );

        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            a1.neq(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.neq(&a2).into_iter().collect_vec(),
            a2_2chunks.neq(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            a1.gt(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.gt(&a2).into_iter().collect_vec(),
            a2_2chunks.gt(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            a1.gt_eq(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.gt_eq(&a2).into_iter().collect_vec(),
            a2_2chunks.gt_eq(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            a1.lt_eq(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.lt_eq(&a2).into_iter().collect_vec(),
            a2_2chunks.lt_eq(&a1).into_iter().collect_vec()
        );

        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            a1.lt(&a2_2chunks).into_iter().collect_vec()
        );
        assert_eq!(
            a1.lt(&a2).into_iter().collect_vec(),
            a2_2chunks.lt(&a1).into_iter().collect_vec()
        );
    }

    #[test]
    fn test_left_right() {
        // This failed with arrow comparisons. TODO: check minimal arrow example with one array being
        // sliced
        let a1: Int32Chunked = (&[Some(1), Some(2)]).iter().copied().collect();
        let a1 = a1.slice(1, 1).unwrap();
        let a2: Int32Chunked = (&[Some(2)]).iter().copied().collect();
        assert_eq!(a1.eq(&a2).sum(), a2.eq(&a1).sum());
        assert_eq!(a1.neq(&a2).sum(), a2.neq(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());

        let a1: Utf8Chunked = (&["a", "b"]).iter().copied().collect();
        let a1 = a1.slice(1, 1).unwrap();
        let a2: Utf8Chunked = (&["b"]).iter().copied().collect();
        assert_eq!(a1.eq(&a2).sum(), a2.eq(&a1).sum());
        assert_eq!(a1.neq(&a2).sum(), a2.neq(&a1).sum());
        assert_eq!(a1.gt(&a2).sum(), a2.gt(&a1).sum());
        assert_eq!(a1.lt(&a2).sum(), a2.lt(&a1).sum());
        assert_eq!(a1.lt_eq(&a2).sum(), a2.lt_eq(&a1).sum());
        assert_eq!(a1.gt_eq(&a2).sum(), a2.gt_eq(&a1).sum());
    }
}
