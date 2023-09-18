use std::cmp::Ordering;

use crate::compute::take;
use crate::error::{Error, Result};
use crate::{
    array::{ord, Array, PrimitiveArray},
    types::Index,
};

use super::{sort_to_indices, SortOptions};
use crate::array::ord::DynComparator;

type IsValid = Box<dyn Fn(usize) -> bool + Send + Sync>;

/// One column to be used in lexicographical sort
#[derive(Clone, Debug)]
pub struct SortColumn<'a> {
    /// The array to sort
    pub values: &'a dyn Array,
    /// The options to apply to the sort
    pub options: Option<SortOptions>,
}

/// Sort a list of [`Array`] using [`SortOptions`] provided for each array.
/// # Implementaqtion
/// The sort is stable and lexicographical on values.
///
/// Returns an [`Error`] if any of the array type is either unsupported by
/// `lexsort_to_indices` or `take`.
///
/// Example:
///
/// ```
/// use std::convert::From;
/// use arrow2::array::{Utf8Array, Int64Array, Array};
/// use arrow2::compute::sort::{SortColumn, SortOptions, lexsort};
/// use arrow2::datatypes::DataType;
///
/// let int64 = Int64Array::from(&[None, Some(-2), Some(89), Some(-64), Some(101)]);
/// let utf8 = Utf8Array::<i32>::from(&vec![Some("hello"), Some("world"), Some(","), Some("foobar"), Some("!")]);
///
/// let sorted_chunk = lexsort::<i32>(&vec![
///     SortColumn {
///         values: &int64,
///         options: None,
///     },
///     SortColumn {
///         values: &utf8,
///         options: Some(SortOptions {
///             descending: true,
///             nulls_first: false,
///         }),
///     },
/// ], None).unwrap();
///
/// let sorted = sorted_chunk[0].as_any().downcast_ref::<Int64Array>().unwrap();
/// assert_eq!(sorted.value(1), -64);
/// assert!(sorted.is_null(0));
/// ```
pub fn lexsort<I: Index>(
    columns: &[SortColumn],
    limit: Option<usize>,
) -> Result<Vec<Box<dyn Array>>> {
    let indices = lexsort_to_indices::<I>(columns, limit)?;
    columns
        .iter()
        .map(|c| take::take(c.values, &indices))
        .collect()
}

#[inline]
fn build_is_valid(array: &dyn Array) -> IsValid {
    if let Some(validity) = array.validity() {
        let validity = validity.clone();
        Box::new(move |x| unsafe { validity.get_bit_unchecked(x) })
    } else {
        Box::new(move |_| true)
    }
}

pub(crate) fn build_compare(array: &dyn Array, sort_option: SortOptions) -> Result<DynComparator> {
    build_compare_impl(array, sort_option, &ord::build_compare)
}

pub(crate) fn build_compare_impl(
    array: &dyn Array,
    sort_option: SortOptions,
    build_compare_fn: &dyn Fn(&dyn Array, &dyn Array) -> Result<DynComparator>,
) -> Result<DynComparator> {
    let is_valid = build_is_valid(array);
    let comparator = build_compare_fn(array, array)?;

    Ok(match (sort_option.descending, sort_option.nulls_first) {
        (true, true) => Box::new(move |i: usize, j: usize| match (is_valid(i), is_valid(j)) {
            (true, true) => match (comparator)(i, j) {
                Ordering::Equal => Ordering::Equal,
                other => other.reverse(),
            },
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        }),
        (false, true) => Box::new(move |i: usize, j: usize| match (is_valid(i), is_valid(j)) {
            (true, true) => match (comparator)(i, j) {
                Ordering::Equal => Ordering::Equal,
                other => other,
            },
            (false, true) => Ordering::Less,
            (true, false) => Ordering::Greater,
            (false, false) => Ordering::Equal,
        }),
        (false, false) => Box::new(move |i: usize, j: usize| match (is_valid(i), is_valid(j)) {
            (true, true) => match (comparator)(i, j) {
                Ordering::Equal => Ordering::Equal,
                other => other,
            },
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => Ordering::Equal,
        }),
        (true, false) => Box::new(move |i: usize, j: usize| match (is_valid(i), is_valid(j)) {
            (true, true) => match (comparator)(i, j) {
                Ordering::Equal => Ordering::Equal,
                other => other.reverse(),
            },
            (false, true) => Ordering::Greater,
            (true, false) => Ordering::Less,
            (false, false) => Ordering::Equal,
        }),
    })
}

/// Sorts a list of [`SortColumn`] into a non-nullable [`PrimitiveArray`]
/// representing the indices that would sort the columns.
pub fn lexsort_to_indices<I: Index>(
    columns: &[SortColumn],
    limit: Option<usize>,
) -> Result<PrimitiveArray<I>> {
    lexsort_to_indices_impl(columns, limit, &ord::build_compare)
}

/// Sorts a list of [`SortColumn`] into a non-nullable [`PrimitiveArray`]
/// representing the indices that would sort the columns.
/// Implementing custom `build_compare_fn` for unsupportd data types.
pub fn lexsort_to_indices_impl<I: Index>(
    columns: &[SortColumn],
    limit: Option<usize>,
    build_compare_fn: &dyn Fn(&dyn Array, &dyn Array) -> Result<DynComparator>,
) -> Result<PrimitiveArray<I>> {
    if columns.is_empty() {
        return Err(Error::InvalidArgumentError(
            "Sort requires at least one column".to_string(),
        ));
    }
    if columns.len() == 1 {
        // fallback to non-lexical sort
        let column = &columns[0];
        if let Ok(indices) =
            sort_to_indices(column.values, &column.options.unwrap_or_default(), limit)
        {
            return Ok(indices);
        }
    }

    let row_count = columns[0].values.len();
    if columns.iter().any(|item| item.values.len() != row_count) {
        return Err(Error::InvalidArgumentError(
            "lexical sort columns have different row counts".to_string(),
        ));
    };

    // map arrays to comparators
    let comparators = columns
        .iter()
        .map(|column| -> Result<DynComparator> {
            build_compare_impl(
                column.values,
                column.options.unwrap_or_default(),
                build_compare_fn,
            )
        })
        .collect::<Result<Vec<DynComparator>>>()?;

    let lex_comparator = |a_idx: &I, b_idx: &I| -> Ordering {
        let a_idx = a_idx.to_usize();
        let b_idx = b_idx.to_usize();
        for comparator in comparators.iter() {
            match comparator(a_idx, b_idx) {
                Ordering::Equal => continue,
                other => return other,
            }
        }

        Ordering::Equal
    };

    let mut values = I::range(0, row_count).unwrap().collect::<Vec<_>>();

    if let Some(limit) = limit {
        let limit = limit.min(row_count);
        let before = if limit < row_count {
            let (before, _, _) = values.select_nth_unstable_by(limit, lex_comparator);
            before
        } else {
            &mut values[..]
        };
        before.sort_unstable_by(lex_comparator);
        values.truncate(limit);
        values.shrink_to_fit();
    } else {
        values.sort_unstable_by(lex_comparator);
    }

    let data_type = I::PRIMITIVE.into();
    Ok(PrimitiveArray::<I>::new(data_type, values.into(), None))
}
