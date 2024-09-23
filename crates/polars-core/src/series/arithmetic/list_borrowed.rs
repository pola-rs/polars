//! Allow arithmetic operations for ListChunked.

use super::*;
use crate::chunked_array::builder::AnonymousListBuilder;

/// Given an ArrayRef with some primitive values, wrap it in list(s) until it
/// matches the requested shape.
fn reshape_list_based_on(data: &ArrayRef, shape: &ArrayRef) -> ArrayRef {
    if let Some(list_chunk) = shape.as_any().downcast_ref::<LargeListArray>() {
        let result = LargeListArray::new(
            list_chunk.dtype().clone(),
            list_chunk.offsets().clone(),
            reshape_list_based_on(data, list_chunk.values()),
            list_chunk.validity().cloned(),
        );
        Box::new(result)
    } else {
        data.clone()
    }
}

/// Given an ArrayRef, return true if it's a LargeListArrays and it has one or
/// more nulls.
fn does_list_have_nulls(data: &ArrayRef) -> bool {
    if let Some(list_chunk) = data.as_any().downcast_ref::<LargeListArray>() {
        if list_chunk
            .validity()
            .map(|bitmap| bitmap.unset_bits() > 0)
            .unwrap_or(false)
        {
            true
        } else {
            does_list_have_nulls(list_chunk.values())
        }
    } else {
        false
    }
}

/// Return whether the left and right have the same shape. We assume neither has
/// any nulls, recursively.
fn lists_same_shapes(left: &ArrayRef, right: &ArrayRef) -> bool {
    debug_assert!(!does_list_have_nulls(left));
    debug_assert!(!does_list_have_nulls(right));
    let left_as_list = left.as_any().downcast_ref::<LargeListArray>();
    let right_as_list = right.as_any().downcast_ref::<LargeListArray>();
    match (left_as_list, right_as_list) {
        (Some(left), Some(right)) => {
            left.offsets() == right.offsets() && lists_same_shapes(left.values(), right.values())
        },
        (None, None) => left.len() == right.len(),
        _ => false,
    }
}

impl ListChunked {
    /// Helper function for NumOpsDispatchInner implementation for ListChunked.
    ///
    /// Run the given `op` on `self` and `rhs`.
    fn arithm_helper(
        &self,
        rhs: &Series,
        op: &dyn Fn(&Series, &Series) -> PolarsResult<Series>,
        has_nulls: Option<bool>,
    ) -> PolarsResult<Series> {
        polars_ensure!(
            self.len() == rhs.len(),
            InvalidOperation: "can only do arithmetic operations on Series of the same size; got {} and {}",
            self.len(),
            rhs.len()
        );

        let mut has_nulls = has_nulls.unwrap_or(false);
        if !has_nulls {
            for chunk in self.chunks().iter() {
                if does_list_have_nulls(chunk) {
                    has_nulls = true;
                    break;
                }
            }
        }
        if !has_nulls {
            for chunk in rhs.chunks().iter() {
                if does_list_have_nulls(chunk) {
                    has_nulls = true;
                    break;
                }
            }
        }
        if has_nulls {
            // A slower implementation since we can't just add the underlying
            // values Arrow arrays. Given nulls, the two values arrays might not
            // line up the way we expect.
            let mut result = AnonymousListBuilder::new(
                self.name().clone(),
                self.len(),
                Some(self.inner_dtype().clone()),
            );
            let combined = self.amortized_iter().zip(rhs.list()?.amortized_iter()).map(|(a, b)| {
                    let (Some(a_owner), Some(b_owner)) = (a, b) else {
                        // Operations with nulls always result in nulls:
                        return Ok(None);
                    };
                    let a = a_owner.as_ref();
                    let b = b_owner.as_ref();
                    polars_ensure!(
                        a.len() == b.len(),
                        InvalidOperation: "can only do arithmetic operations on lists of the same size; got {} and {}",
                        a.len(),
                        b.len()
                    );
                    let chunk_result = if let Ok(a_listchunked) = a.list() {
                        // If `a` contains more lists, we're going to reach this
                        // function recursively, and again have to decide whether to
                        // use the fast path (no nulls) or slow path (there were
                        // nulls). Since we know there were nulls, that means we
                        // have to stick to the slow path, so pass that information
                        // along.
                        a_listchunked.arithm_helper(b, op, Some(true))
                    } else {
                        op(a, b)
                    };
                    chunk_result.map(Some)
                }).collect::<PolarsResult<Vec<Option<Series>>>>()?;
            for s in combined.iter() {
                if let Some(s) = s {
                    result.append_series(s)?;
                } else {
                    result.append_null();
                }
            }
            return Ok(result.finish().into());
        }
        let l_rechunked = self.clone().rechunk().into_series();
        let l_leaf_array = l_rechunked.get_leaf_array();
        let r_leaf_array = rhs.rechunk().get_leaf_array();
        polars_ensure!(
            lists_same_shapes(&l_leaf_array.chunks()[0], &r_leaf_array.chunks()[0]),
            InvalidOperation: "can only do arithmetic operations on lists of the same size"
        );

        let result = op(&l_leaf_array, &r_leaf_array)?;

        // We now need to wrap the Arrow arrays with the metadata that turns
        // them into lists:
        // TODO is there a way to do this without cloning the underlying data?
        let result_chunks = result.chunks();
        assert_eq!(result_chunks.len(), 1);
        let left_chunk = &l_rechunked.chunks()[0];
        let result_chunk = reshape_list_based_on(&result_chunks[0], left_chunk);

        unsafe {
            let mut result =
                ListChunked::new_with_dims(self.field.clone(), vec![result_chunk], 0, 0);
            result.compute_len();
            Ok(result.into())
        }
    }
}

impl NumOpsDispatchInner for ListType {
    fn add_to(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        lhs.arithm_helper(rhs, &|l, r| l.add_to(r), None)
    }
    fn subtract(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        lhs.arithm_helper(rhs, &|l, r| l.subtract(r), None)
    }
    fn multiply(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        lhs.arithm_helper(rhs, &|l, r| l.multiply(r), None)
    }
    fn divide(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        lhs.arithm_helper(rhs, &|l, r| l.divide(r), None)
    }
    fn remainder(lhs: &ListChunked, rhs: &Series) -> PolarsResult<Series> {
        lhs.arithm_helper(rhs, &|l, r| l.remainder(r), None)
    }
}
