use polars_error::PolarsResult;

use super::{BooleanChunked, ChunkCompareEq, ChunkCompareIneq, ChunkExpandAtIndex, Column, Series};

macro_rules! column_element_wise_broadcasting {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        match ($lhs, $rhs) {
            (Column::Series(lhs), Column::Series(rhs)) => $op(lhs, rhs),
            (Column::Series(lhs), Column::Scalar(rhs)) => $op(lhs, &rhs.as_single_value_series()),
            (Column::Scalar(lhs), Column::Series(rhs)) => $op(&lhs.as_single_value_series(), rhs),
            (Column::Scalar(lhs), Column::Scalar(rhs)) => {
                $op(&lhs.as_single_value_series(), &rhs.as_single_value_series()).map(|ca| {
                    if ca.len() == 0 {
                        ca
                    } else {
                        ca.new_from_index(0, lhs.len())
                    }
                })
            },
        }
    };
}

impl ChunkCompareEq<&Column> for Column {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    #[inline]
    fn equal(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(self, rhs, <Series as ChunkCompareEq<&Series>>::equal)
    }

    /// Create a boolean mask by checking for equality.
    #[inline]
    fn equal_missing(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(
            self,
            rhs,
            <Series as ChunkCompareEq<&Series>>::equal_missing
        )
    }

    /// Create a boolean mask by checking for inequality.
    #[inline]
    fn not_equal(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(self, rhs, <Series as ChunkCompareEq<&Series>>::not_equal)
    }

    /// Create a boolean mask by checking for inequality.
    #[inline]
    fn not_equal_missing(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(
            self,
            rhs,
            <Series as ChunkCompareEq<&Series>>::not_equal_missing
        )
    }
}

impl ChunkCompareIneq<&Column> for Column {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking if self > rhs.
    #[inline]
    fn gt(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(self, rhs, <Series as ChunkCompareIneq<&Series>>::gt)
    }

    /// Create a boolean mask by checking if self >= rhs.
    #[inline]
    fn gt_eq(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(self, rhs, <Series as ChunkCompareIneq<&Series>>::gt_eq)
    }

    /// Create a boolean mask by checking if self < rhs.
    #[inline]
    fn lt(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(self, rhs, <Series as ChunkCompareIneq<&Series>>::lt)
    }

    /// Create a boolean mask by checking if self <= rhs.
    #[inline]
    fn lt_eq(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        column_element_wise_broadcasting!(self, rhs, <Series as ChunkCompareIneq<&Series>>::lt_eq)
    }
}
