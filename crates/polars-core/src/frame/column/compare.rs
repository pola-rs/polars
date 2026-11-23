use polars_arrow::array::{BooleanArray, PrimitiveArray};
use polars_arrow::bitmap::Bitmap;
use polars_arrow::datatypes::ArrowDataType;
use polars_compute::comparisons::TotalEqKernel;
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
    // Without a null result for a value, the result is only null on the null rows.
    let valid_values = valid.as_deref().filter(|valid| !valid[..n].iter().all(|v| *v));

    let chunks = column.codes().downcast_iter().map(|codes| {
        let has_nulls = codes.validity().is_some_and(|v| v.unset_bits() > 0);
        let values = lookup_bits(codes, &bits);
        let validity = match (valid_values, valid.as_deref()) {
            (Some(valid), _) => Some(lookup_bits(codes, valid)),
            (None, Some(valid)) if has_nulls && !valid[n] => codes.validity().cloned(),
            _ => None,
        };
        BooleanArray::new(ArrowDataType::Boolean, values, validity)
    });
    Ok(BooleanChunked::from_chunk_iter(name, chunks))
}

/// `table[code]` for every row of `codes`, and `table[n]` (the entry for a null) for its null
/// rows, where `table` has an entry for each of the `n` dictionary values and one for a null.
fn lookup_bits(codes: &PrimitiveArray<u32>, table: &[bool]) -> Bitmap {
    let n = table.len() - 1;
    let mut matching = table[..n].iter().enumerate().filter(|(_, b)| **b);
    let first = matching.next();
    let bits = match (first, matching.next()) {
        (None, _) => Bitmap::new_zeroed(codes.len()),
        // Comparing the codes with one code takes a fraction of the time of a lookup per row.
        (Some((code, _)), None) => codes.tot_eq_kernel_broadcast(&(code as u32)),
        // The code of a null row can be anything; its result is replaced below.
        _ => Bitmap::from_trusted_len_iter(
            codes
                .values()
                .iter()
                .map(|c| unsafe { *table.get_unchecked((*c as usize).min(n)) }),
        ),
    };
    match codes.validity().filter(|v| v.unset_bits() > 0) {
        Some(validity) if table[n] => &bits | &!validity,
        Some(validity) => &bits & validity,
        None => bits,
    }
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
