use num_traits::ToPrimitive;
use polars_core::error::{PolarsResult, polars_bail, polars_ensure};
use polars_core::prelude::{ChunkedArray, DataType, IdxCa, IdxSize, PolarsIntegerType, Series};

/// UNSIGNED conversion:
///
/// - `0 <= v < target_len`  → `Some(v as IdxSize)`
/// - `v >= target_len`      → `None`
/// - values that cannot be represented as `u64` → `None`
fn convert_unsigned<T>(ca: &ChunkedArray<T>, target_len: usize) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    T::Native: ToPrimitive,
{
    let len_u64 = target_len as u64;

    let out: IdxCa = ca
        .into_iter()
        .map(|opt_v| {
            opt_v.and_then(|v| {
                let v_u64 = match v.to_u64() {
                    Some(x) => x,
                    None => return None, // cannot represent → treat as OOB
                };

                if v_u64 < len_u64 {
                    Some(v_u64 as IdxSize)
                } else {
                    None
                }
            })
        })
        .collect();

    Ok(out)
}

/// SIGNED conversion with Python-style negative semantics:
///
/// - `0 <= v < target_len`          → `Some(v as IdxSize)`
/// - `v >= target_len`              → `None`
/// - `-target_len <= v < 0`         → `Some(target_len + v)`
/// - `v < -target_len`              → `None`
/// - values that cannot be represented as `i64` → `None`
fn convert_signed<T>(ca: &ChunkedArray<T>, target_len: usize) -> PolarsResult<IdxCa>
where
    T: PolarsIntegerType,
    T::Native: ToPrimitive,
{
    let len_i64 = target_len as i64;

    let out: IdxCa = ca
        .into_iter()
        .map(|opt_v| {
            opt_v.and_then(|v| {
                let v_i64 = match v.to_i64() {
                    Some(x) => x,
                    None => return None, // cannot represent → treat as OOB
                };

                if v_i64 >= 0 {
                    // 0 <= v < len
                    if v_i64 < len_i64 {
                        Some(v_i64 as IdxSize)
                    } else {
                        None
                    }
                } else {
                    // negative index: valid iff -len <= v < 0
                    if v_i64 >= -len_i64 {
                        let pos = len_i64 + v_i64; // in [0, len)
                        debug_assert!(pos >= 0 && pos < len_i64);
                        Some(pos as IdxSize)
                    } else {
                        None
                    }
                }
            })
        })
        .collect();

    Ok(out)
}

/// Convert arbitrary integer Series into IdxCa, using `target_len` as logical length.
///
/// - All OOB indices are mapped to null in `convert_*`.
/// - We track null counts before and after:
///   - if `null_on_oob == true`, extra nulls are expected and we just return.
///   - if `null_on_oob == false` and new nulls appear, we raise OutOfBounds.
pub fn convert_to_unsigned_index(
    s: &Series,
    target_len: usize,
    null_on_oob: bool,
) -> PolarsResult<IdxCa> {
    let dtype = s.dtype();
    polars_ensure!(
        dtype.is_integer(),
        InvalidOperation: "expected integers as index"
    );

    assert!(
        (target_len as u128) <= (IdxSize::MAX as u128),
        "internal error: target_len does not fit in IdxSize"
    );

    let in_nulls = s.null_count();

    // Normalize to IdxCa with all OOB indices already mapped to nulls.
    let idx: IdxCa = match dtype {
        // ----- SIGNED -----
        DataType::Int64 => {
            let ca = s.i64().unwrap();
            convert_signed(ca, target_len)?
        },
        DataType::Int32 => {
            let ca = s.i32().unwrap();
            convert_signed(ca, target_len)?
        },
        #[cfg(feature = "dtype-i16")]
        DataType::Int16 => {
            let ca = s.i16().unwrap();
            convert_signed(ca, target_len)?
        },
        #[cfg(feature = "dtype-i8")]
        DataType::Int8 => {
            let ca = s.i8().unwrap();
            convert_signed(ca, target_len)?
        },
        #[cfg(feature = "dtype-i128")]
        DataType::Int128 => {
            let ca = s.i128().unwrap();
            convert_signed(ca, target_len)?
        },

        // ----- UNSIGNED -----
        DataType::UInt64 => {
            let ca = s.u64().unwrap();
            convert_unsigned(ca, target_len)?
        },
        DataType::UInt32 => {
            let ca = s.u32().unwrap();
            convert_unsigned(ca, target_len)?
        },
        #[cfg(feature = "dtype-u16")]
        DataType::UInt16 => {
            let ca = s.u16().unwrap();
            convert_unsigned(ca, target_len)?
        },
        #[cfg(feature = "dtype-u8")]
        DataType::UInt8 => {
            let ca = s.u8().unwrap();
            convert_unsigned(ca, target_len)?
        },
        #[cfg(feature = "dtype-u128")]
        DataType::UInt128 => {
            let ca = s.u128().unwrap();
            convert_unsigned(ca, target_len)?
        },

        _ => unreachable!(),
    };

    let out_nulls = idx.null_count();

    if !null_on_oob && out_nulls > in_nulls {
        polars_bail!(
            OutOfBounds: "gather indices are out of bounds"
        );
    }

    Ok(idx)
}
