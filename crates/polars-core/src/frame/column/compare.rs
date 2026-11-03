use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::pl_str::PlSmallStr;

use super::{
    BooleanChunked, ChunkCompareEq, ChunkCompareIneq, ChunkExpandAtIndex, Column, DictionaryColumn,
    IntoSeries, Series,
};
use crate::prelude::{ChunkFullNull, StringChunked};

/// `op` evaluated on each used value of `column` and spread to its rows.
fn dictionary_mask(
    column: &DictionaryColumn,
    op: impl Fn(&Series) -> PolarsResult<BooleanChunked>,
) -> PolarsResult<BooleanChunked> {
    if column.dict().len() <= column.len() {
        match dictionary_lookup_mask(column, &op) {
            Ok(mask) => return Ok(mask),
            Err(err) => {
                // A value no row uses may fail where the used ones don't; that falls back to the
                // used ones. An error without any values (e.g. of the dtypes) is returned at once.
                let empty = StringChunked::full_null(column.name().clone(), 0).into_series();
                if op(&empty).is_err() {
                    return Err(err);
                }
            },
        }
    }
    let out = column.apply_elementwise(|values| op(values).map(|ca| ca.into_series()))?;
    Ok(out.bool()?.clone())
}

/// `op` evaluated once on every dictionary value and on a null, then looked up per row by code.
fn dictionary_lookup_mask(
    column: &DictionaryColumn,
    op: impl Fn(&Series) -> PolarsResult<BooleanChunked>,
) -> PolarsResult<BooleanChunked> {
    let n = column.dict().len();
    let mut values =
        StringChunked::with_chunk(column.name().clone(), column.dict().values().clone());
    values.append(&StringChunked::full_null(PlSmallStr::EMPTY, 1))?;
    let table = op(&values.into_series())?;
    let table = table.rechunk();
    polars_ensure!(
        table.len() == n + 1,
        ComputeError: "comparison changed the length of a dictionary column"
    );
    let name = table.name().clone();
    let table = table.downcast_as_array();
    let bits: Vec<bool> = table.values().iter().collect();
    let valid: Option<Vec<bool>> = table.validity().map(|v| v.iter().collect());

    let codes = column.codes().rechunk();
    let codes = codes.downcast_as_array();
    // Null rows look up the last entry, the result for a null.
    let slots = codes.values().iter().enumerate().map(|(i, c)| {
        // SAFETY: valid codes are below `n` (invariant).
        if codes
            .validity()
            .is_none_or(|v| unsafe { v.get_bit_unchecked(i) })
        {
            *c as usize
        } else {
            n
        }
    });
    let (values, validity) = match &valid {
        None if codes.validity().is_none() => (
            // SAFETY: every code is below `n` (invariant) and `bits` has `n + 1` entries.
            Bitmap::from_trusted_len_iter(
                codes
                    .values()
                    .iter()
                    .map(|c| unsafe { *bits.get_unchecked(*c as usize) }),
            ),
            None,
        ),
        None => (
            Bitmap::from_trusted_len_iter(slots.map(|s| unsafe { *bits.get_unchecked(s) })),
            None,
        ),
        Some(valid) => {
            let slots: Vec<usize> = slots.collect();
            (
                Bitmap::from_trusted_len_iter(
                    slots.iter().map(|s| unsafe { *bits.get_unchecked(*s) }),
                ),
                Some(Bitmap::from_trusted_len_iter(
                    slots.iter().map(|s| unsafe { *valid.get_unchecked(*s) }),
                )),
            )
        },
    };
    let arr = BooleanArray::new(ArrowDataType::Boolean, values, validity);
    Ok(BooleanChunked::with_chunk(name, arr))
}

macro_rules! column_element_wise_broadcasting {
    ($lhs:expr, $rhs:expr, $op:expr) => {
        match ($lhs, $rhs) {
            // Compare each used dictionary value with the constant once. Only when the result has
            // the dictionary column's length: a length-1 one broadcasts to a longer constant.
            (Column::Dictionary(lhs), rhs)
                if rhs.len() == 1
                    || (rhs.len() == lhs.len() && rhs.as_scalar_column().is_some()) =>
            {
                let rhs = rhs.as_materialized_series_maintain_scalar();
                dictionary_mask(lhs, |values| $op(values, &rhs))
            },
            (lhs, Column::Dictionary(rhs))
                if lhs.len() == 1
                    || (lhs.len() == rhs.len() && lhs.as_scalar_column().is_some()) =>
            {
                let lhs = lhs.as_materialized_series_maintain_scalar();
                dictionary_mask(rhs, |values| $op(&lhs, values))
            },
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
            (lhs, rhs) => $op(lhs.as_materialized_series(), rhs.as_materialized_series()),
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
