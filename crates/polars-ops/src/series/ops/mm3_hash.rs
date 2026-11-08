use polars_core::chunked_array::ops::arity::unary_elementwise_values;
use polars_core::prelude::*;

/// 32-bit MurmurHash3 (x86 variant).
pub fn murmur3_x86_32(bytes: &[u8], seed: u32) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;

    let mut h = seed;
    let mut chunks = bytes.chunks_exact(4);

    for chunk in chunks.by_ref() {
        let mut k = u32::from_le_bytes(chunk.try_into().unwrap());
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }

    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut k = 0u32;
        for (i, b) in tail.iter().enumerate() {
            k |= (*b as u32) << (8 * i);
        }
        k = k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);
        h ^= k;
    }

    h ^= bytes.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^= h >> 16;
    h
}

#[inline]
fn hash_bytes(bytes: &[u8]) -> i32 {
    murmur3_x86_32(bytes, 0) as i32
}

#[inline]
fn hash_long(v: i64) -> i32 {
    hash_bytes(&v.to_le_bytes())
}

/// Hash the minimum-length big-endian two's-complement representation of the unscaled value.
#[cfg(feature = "dtype-decimal")]
#[inline]
fn hash_decimal(v: i128) -> i32 {
    let magnitude = if v < 0 { !v } else { v };
    let num_bits = 128 - magnitude.leading_zeros() as usize;
    let num_bytes = num_bits / 8 + 1;
    hash_bytes(&v.to_be_bytes()[16 - num_bytes..])
}

fn hash_i64_values(ca: &Int64Chunked, f: impl Fn(i64) -> i64) -> Int32Chunked {
    unary_elementwise_values(ca, |v| hash_long(f(v)))
}

/// Hash values using the 32-bit MurmurHash3 as defined by the Iceberg specification (Appendix B).
///
/// Integers, dates, times and timestamps are hashed as 8-byte little-endian longs (times and
/// timestamps in microseconds), decimals as the minimal big-endian two's-complement bytes of the
/// unscaled value, and strings and binary as their bytes.
pub fn mm3_hash(s: &Series) -> PolarsResult<Series> {
    use DataType as D;

    let name = s.name().clone();

    let out: Int32Chunked = match s.dtype() {
        D::Null => Int32Chunked::full_null(name.clone(), s.len()),
        dt if dt.is_integer() => {
            let s = s.strict_cast(&D::Int64)?;
            hash_i64_values(s.i64()?, |v| v)
        },
        #[cfg(feature = "dtype-date")]
        D::Date => {
            let s = s.to_physical_repr().cast(&D::Int64)?;
            hash_i64_values(s.i64()?, |v| v)
        },
        #[cfg(feature = "dtype-time")]
        D::Time => {
            let ca = s.time()?.physical();
            hash_i64_values(ca, |ns| ns.div_euclid(1_000))
        },
        #[cfg(feature = "dtype-datetime")]
        D::Datetime(tu, _) => {
            let ca = s.datetime()?.physical();
            match tu {
                TimeUnit::Nanoseconds => hash_i64_values(ca, |ns| ns.div_euclid(1_000)),
                TimeUnit::Microseconds => hash_i64_values(ca, |us| us),
                TimeUnit::Milliseconds => {
                    let ca = ca.try_apply_nonnull_values_generic::<Int64Type, _, _, _>(|ms| {
                        ms.checked_mul(1_000).ok_or_else(|| {
                            polars_err!(ComputeError: "datetime out of range for microsecond precision: {ms}ms")
                        })
                    })?;
                    hash_i64_values(&ca, |us| us)
                },
            }
        },
        #[cfg(feature = "dtype-decimal")]
        D::Decimal(_, _) => unary_elementwise_values(s.decimal()?.physical(), hash_decimal),
        D::String => unary_elementwise_values(s.str()?, |v: &str| hash_bytes(v.as_bytes())),
        D::Binary => unary_elementwise_values(s.binary()?, hash_bytes),
        dt => polars_bail!(InvalidOperation: "mm3_hash not supported for dtype {dt}"),
    };

    Ok(out.with_name(name).into_series())
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_murmur3_x86_32() {
        // Reference values from the Iceberg specification (Appendix B).
        assert_eq!(hash_long(34), 2017239379);
        assert_eq!(hash_bytes("iceberg".as_bytes()), 1210000089);
        assert_eq!(hash_bytes(&[0, 1, 2, 3]), -188683207);
        #[cfg(feature = "dtype-decimal")]
        assert_eq!(hash_decimal(1420), -500754589);
    }
}
