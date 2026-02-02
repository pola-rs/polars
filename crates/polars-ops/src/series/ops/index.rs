use arrow::array::Array;
use arrow::bitmap::BitmapBuilder;
use arrow::compute::utils::combine_validities_and;
use arrow::datatypes::IdxArr;
use num_traits::{Bounded, ToPrimitive, Zero};
use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::prelude::{ChunkedArray, IdxCa, IdxSize, PolarsIntegerType, Series};
use polars_core::with_match_physical_integer_polars_type;
use polars_utils::select::select_unpredictable;
use polars_utils::vec::PushUnchecked;

/// UNSIGNED conversion:
/// - `0 <= v < target_len`  → `Some(v)`
/// - `v >= target_len`      → `None`
///
/// SIGNED conversion with Python-style negative semantics:
/// - `v < -target_len`              → `None`
/// - `-target_len <= v < 0`         → `Some(target_len + v)`
/// - `0 <= v < target_len`          → `Some(v)`
/// - `v >= target_len`              → `None`
pub fn convert_and_bound_idx_ca<T>(
    ca: &ChunkedArray<T>,
    target_len: usize,
    null_on_oob: bool,
) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    T::Native: ToPrimitive,
{
    let mut out = Vec::with_capacity(ca.len());
    let mut in_bounds = BitmapBuilder::with_capacity(ca.len());
    assert!(target_len < IdxSize::MAX as usize);

    let unsigned = T::Native::min_value() == T::Native::zero(); // Optimized to constant by compiler.
    if unsigned {
        let len_u64 = target_len as u64;
        for arr in ca.downcast_iter() {
            for v in arr.values().iter() {
                // SAFETY: we reserved.
                unsafe {
                    if let Some(v_u64) = v.to_u64() {
                        // Usually infallible.
                        out.push_unchecked(v_u64 as IdxSize);
                        in_bounds.push_unchecked(v_u64 < len_u64);
                    } else {
                        in_bounds.push_unchecked(false);
                    }
                }
            }
        }
    } else {
        let len_i64 = target_len as i64;
        for arr in ca.downcast_iter() {
            for v in arr.values().iter() {
                // SAFETY: we reserved.
                unsafe {
                    if let Some(v_i64) = v.to_i64() {
                        // Usually infallible.
                        let mut shifted = v_i64;
                        shifted += select_unpredictable(v_i64 < 0, len_i64, 0);
                        out.push_unchecked(shifted as IdxSize);
                        in_bounds.push_unchecked((v_i64 >= -len_i64) & (v_i64 < len_i64));
                    } else {
                        in_bounds.push_unchecked(false);
                    }
                }
            }
        }
    }

    let idx_arr = IdxArr::from_vec(out);
    let in_bounds_valid = in_bounds.into_opt_validity();
    let ca_valid = ca.rechunk_validity();
    let valid = combine_validities_and(in_bounds_valid.as_ref(), ca_valid.as_ref());
    let out = idx_arr.with_validity(valid);

    if !null_on_oob && out.null_count() != ca.null_count() {
        polars_bail!(
            OutOfBounds: "gather indices are out of bounds"
        );
    }

    Ok(out.into())
}

/// Convert arbitrary integer Series into IdxCa, using `target_len` as logical length.
///
/// - All OOB indices are mapped to null in `convert_*`.
/// - We track null counts before and after:
///   - if `null_on_oob == true`, extra nulls are expected and we just return.
///   - if `null_on_oob == false` and new nulls appear, we raise OutOfBounds.
pub fn convert_and_bound_index(
    s: &Series,
    target_len: usize,
    null_on_oob: bool,
) -> PolarsResult<IdxCa> {
    let dtype = s.dtype();
    polars_ensure!(
        dtype.is_integer(),
        InvalidOperation: "expected integers as index"
    );

    with_match_physical_integer_polars_type!(dtype, |$T| {
        let ca: &ChunkedArray<$T> = s.as_ref().as_ref();
        convert_and_bound_idx_ca(ca, target_len, null_on_oob)
    })
}
