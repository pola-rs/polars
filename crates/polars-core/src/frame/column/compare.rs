use polars_arrow::array::BooleanArray;
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_error::{PolarsResult, polars_ensure};
use polars_utils::broadcast::broadcast_len;
use polars_utils::pl_str::PlSmallStr;

use super::{
    BooleanChunked, ChunkCompareEq, ChunkCompareIneq, ChunkExpandAtIndex, Column, DictionaryColumn,
    IntoSeries, Series,
};
use crate::prelude::{ChunkFullNull, StringChunked};

/// `op` evaluated on each used value of `column` and spread to its rows.
fn dictionary_mask(
    column: &DictionaryColumn,
    op: &dyn Fn(&Series) -> PolarsResult<BooleanChunked>,
) -> PolarsResult<BooleanChunked> {
    if column.dict().len() <= column.len() {
        match dictionary_lookup_mask(column, op) {
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
    op: &dyn Fn(&Series) -> PolarsResult<BooleanChunked>,
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
    // SAFETY: valid codes are below `n` (invariant) and the tables have `n + 1` entries.
    let (values, validity) = match codes.validity().filter(|v| v.unset_bits() > 0) {
        None => {
            let gather = |table: &[bool]| {
                Bitmap::from_trusted_len_iter(
                    codes
                        .values()
                        .iter()
                        .map(|c| unsafe { *table.get_unchecked(*c as usize) }),
                )
            };
            let validity = valid
                .as_deref()
                .filter(|valid| !valid[..n].iter().all(|v| *v))
                .map(gather);
            (gather(&bits), validity)
        },
        Some(code_validity) => {
            // Null rows look up the last entry, the result for a null.
            let gather = |table: &[bool]| {
                Bitmap::from_trusted_len_iter(codes.values().iter().zip(code_validity).map(
                    |(c, is_valid)| unsafe {
                        *table.get_unchecked(if is_valid { *c as usize } else { n })
                    },
                ))
            };
            (gather(&bits), valid.as_deref().map(gather))
        },
    };
    let arr = BooleanArray::new(ArrowDataType::Boolean, values, validity);
    Ok(BooleanChunked::with_chunk(name, arr))
}

type CompareFn = fn(&Series, &Series) -> PolarsResult<BooleanChunked>;

fn compare_broadcasting(lhs: &Column, rhs: &Column, op: CompareFn) -> PolarsResult<BooleanChunked> {
    match (lhs, rhs) {
        // Compare each used dictionary value with the constant once. Only when the result has
        // the dictionary column's length: a length-1 one broadcasts to a longer constant.
        (Column::Dictionary(lhs), rhs)
            if rhs.len() == 1 || (rhs.len() == lhs.len() && rhs.as_scalar_column().is_some()) =>
        {
            let rhs = rhs.as_materialized_series_maintain_scalar();
            dictionary_mask(lhs, &|values| op(values, &rhs))
        },
        (lhs, Column::Dictionary(rhs))
            if lhs.len() == 1 || (lhs.len() == rhs.len() && lhs.as_scalar_column().is_some()) =>
        {
            let lhs = lhs.as_materialized_series_maintain_scalar();
            dictionary_mask(rhs, &|values| op(&lhs, values))
        },
        (Column::Series(lhs), Column::Scalar(rhs)) => op(lhs, &rhs.as_single_value_series()),
        (Column::Scalar(lhs), Column::Series(rhs)) => op(&lhs.as_single_value_series(), rhs),
        (Column::Scalar(lhs), Column::Scalar(rhs)) => {
            let length = broadcast_len([lhs, rhs])?;
            op(&lhs.as_single_value_series(), &rhs.as_single_value_series()).map(|ca| {
                if ca.is_empty() {
                    ca
                } else {
                    ca.new_from_index(0, length)
                }
            })
        },
        (lhs, rhs) => op(lhs.as_materialized_series(), rhs.as_materialized_series()),
    }
}

impl ChunkCompareEq<&Column> for Column {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking for equality.
    fn equal(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareEq<&Series>>::equal(l, r)
        })
    }

    /// Create a boolean mask by checking for equality.
    fn equal_missing(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareEq<&Series>>::equal_missing(l, r)
        })
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareEq<&Series>>::not_equal(l, r)
        })
    }

    /// Create a boolean mask by checking for inequality.
    fn not_equal_missing(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareEq<&Series>>::not_equal_missing(l, r)
        })
    }
}

impl ChunkCompareIneq<&Column> for Column {
    type Item = PolarsResult<BooleanChunked>;

    /// Create a boolean mask by checking if self > rhs.
    fn gt(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareIneq<&Series>>::gt(l, r)
        })
    }

    /// Create a boolean mask by checking if self >= rhs.
    fn gt_eq(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareIneq<&Series>>::gt_eq(l, r)
        })
    }

    /// Create a boolean mask by checking if self < rhs.
    fn lt(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareIneq<&Series>>::lt(l, r)
        })
    }

    /// Create a boolean mask by checking if self <= rhs.
    fn lt_eq(&self, rhs: &Column) -> PolarsResult<BooleanChunked> {
        compare_broadcasting(self, rhs, |l: &Series, r: &Series| {
            <Series as ChunkCompareIneq<&Series>>::lt_eq(l, r)
        })
    }
}
