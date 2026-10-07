#[cfg(any(
    feature = "binary_encoding",
    all(feature = "strings", feature = "string_encoding")
))]
use base64::Engine as _;
#[cfg(any(
    feature = "binary_encoding",
    all(feature = "strings", feature = "string_encoding")
))]
use base64::engine::general_purpose;
use memchr::memmem::find;
#[cfg(feature = "binary_encoding")]
use polars_arrow::array::{Array, MutablePlBinary};
use polars_compute::cast::{binview_to_fixed_size_list_dyn, binview_to_primitive_dyn};
use polars_compute::size::binary_size_bytes;
use polars_core::prelude::arity::{broadcast_binary_elementwise_values, unary_elementwise_values};

use super::*;

#[cfg(any(
    feature = "binary_encoding",
    all(feature = "strings", feature = "string_encoding")
))]
pub(crate) fn binary_to_hex(ca: &BinaryChunked) -> StringChunked {
    ca.apply_into_string_amortized(|s, buf| {
        // SAFETY: hex output is ASCII.
        let out = unsafe { buf.as_mut_vec() };
        out.resize(s.len() * 2, 0);
        hex::encode_to_slice(s, out).unwrap();
    })
}

#[cfg(any(
    feature = "binary_encoding",
    all(feature = "strings", feature = "string_encoding")
))]
pub(crate) fn binary_to_base64(ca: &BinaryChunked) -> StringChunked {
    ca.apply_into_string_amortized(|s, buf| general_purpose::STANDARD.encode_string(s, buf))
}

/// Decode every value into a reused buffer. `decode` returns `false` on invalid input.
#[cfg(feature = "binary_encoding")]
fn decode_amortized(
    ca: &BinaryChunked,
    strict: bool,
    encoding: &str,
    mut decode: impl FnMut(&[u8], &mut Vec<u8>) -> bool,
) -> PolarsResult<BinaryChunked> {
    let mut buf = Vec::new();
    let chunks = ca.downcast_iter().map(|arr| {
        let mut out = MutablePlBinary::with_capacity(arr.len());
        for opt_s in arr.iter() {
            match opt_s {
                None => out.push_null(),
                Some(s) => {
                    buf.clear();
                    if decode(s, &mut buf) {
                        out.push_value(&buf);
                    } else if strict {
                        polars_bail!(
                            ComputeError:
                            "invalid `{encoding}` encoding found; try setting `strict=false` to ignore"
                        );
                    } else {
                        out.push_null();
                    }
                },
            }
        }
        Ok(out.freeze())
    });
    ChunkedArray::try_from_chunk_iter(ca.name().clone(), chunks)
}

pub trait BinaryNameSpaceImpl: AsBinary {
    /// Slice the binary values.
    ///
    /// Determines a slice starting from `offset` and with length `length` of each of the elements.
    /// `offset` can be negative, in which case the start counts from the end of the bytes.
    fn bin_slice(&self, offset: &Column, length: &Column) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        let offset = offset.cast(&DataType::Int64)?;
        let length = length.strict_cast(&DataType::UInt64)?;

        Ok(super::slice::slice(ca, offset.i64()?, length.u64()?))
    }
    /// Slice the first `n` bytes of the binary value.
    ///
    /// Determines a slice starting at the beginning of the binary data up to offset `n` of each
    /// element. `n` can be negative, in which case the slice ends `n` bytes from the end.
    fn bin_head(&self, n: &Column) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        let n = n.strict_cast(&DataType::Int64)?;

        super::slice::head(ca, n.i64()?)
    }

    /// Slice the last `n` bytes of the binary value.
    ///
    /// Determines a slice starting at offset `n` of each element. `n` can be
    /// negative, in which case the slice begins `n` bytes from the start.
    fn bin_tail(&self, n: &Column) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        let n = n.strict_cast(&DataType::Int64)?;

        super::slice::tail(ca, n.i64()?)
    }

    /// Check if binary contains given literal
    fn contains(&self, lit: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| find(s, lit).is_some();
        unary_elementwise_values(ca, f)
    }

    fn contains_chunked(&self, lit: &BinaryChunked) -> PolarsResult<BooleanChunked> {
        let ca = self.as_binary();
        Ok(match lit.len() {
            1 => match lit.get(0) {
                Some(lit) => ca.contains(lit),
                None => BooleanChunked::full_null(ca.name().clone(), ca.len()),
            },
            _ => {
                polars_ensure!(
                    ca.len() == lit.len() || ca.len() == 1,
                    length_mismatch = "bin.contains",
                    ca.len(),
                    lit.len()
                );
                broadcast_binary_elementwise_values(ca, lit, |src, lit| find(src, lit).is_some())
            },
        })
    }

    /// Check if strings ends with a substring
    fn ends_with(&self, sub: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| s.ends_with(sub);
        ca.apply_nonnull_values_generic(DataType::Boolean, f)
    }

    /// Check if strings starts with a substring
    fn starts_with(&self, sub: &[u8]) -> BooleanChunked {
        let ca = self.as_binary();
        let f = |s: &[u8]| s.starts_with(sub);
        ca.apply_nonnull_values_generic(DataType::Boolean, f)
    }

    fn starts_with_chunked(&self, prefix: &BinaryChunked) -> PolarsResult<BooleanChunked> {
        let ca = self.as_binary();
        Ok(match prefix.len() {
            1 => match prefix.get(0) {
                Some(s) => self.starts_with(s),
                None => BooleanChunked::full_null(ca.name().clone(), ca.len()),
            },
            _ => {
                polars_ensure!(
                    ca.len() == prefix.len() || ca.len() == 1,
                    length_mismatch = "bin.starts_with",
                    ca.len(),
                    prefix.len()
                );
                broadcast_binary_elementwise_values(ca, prefix, |s, sub| s.starts_with(sub))
            },
        })
    }

    fn ends_with_chunked(&self, suffix: &BinaryChunked) -> PolarsResult<BooleanChunked> {
        let ca = self.as_binary();
        Ok(match suffix.len() {
            1 => match suffix.get(0) {
                Some(s) => self.ends_with(s),
                None => BooleanChunked::full_null(ca.name().clone(), ca.len()),
            },
            _ => {
                polars_ensure!(
                    ca.len() == suffix.len() || ca.len() == 1,
                    length_mismatch = "bin.ends_with",
                    ca.len(),
                    suffix.len()
                );
                broadcast_binary_elementwise_values(ca, suffix, |s, sub| s.ends_with(sub))
            },
        })
    }

    /// Get the size of the binary values in bytes.
    fn size_bytes(&self) -> UInt32Chunked {
        let ca = self.as_binary();
        ca.apply_kernel_cast(&binary_size_bytes)
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        decode_amortized(self.as_binary(), strict, "hex", |s, buf| {
            buf.resize(s.len() / 2, 0);
            hex::decode_to_slice(s, buf).is_ok()
        })
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_encode(&self) -> Series {
        binary_to_hex(self.as_binary()).into_series()
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        decode_amortized(self.as_binary(), strict, "base64", |s, buf| {
            general_purpose::STANDARD.decode_vec(s, buf).is_ok()
        })
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_encode(&self) -> Series {
        binary_to_base64(self.as_binary()).into_series()
    }

    #[cfg(feature = "binary_encoding")]
    fn reinterpret(&self, dtype: &DataType, is_little_endian: bool) -> PolarsResult<Series> {
        unsafe {
            Ok(Series::from_chunks_and_dtype_unchecked(
                self.as_binary().name().clone(),
                self._reinterpret_inner(dtype, is_little_endian)?,
                dtype,
            ))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn _reinterpret_inner(
        &self,
        dtype: &DataType,
        is_little_endian: bool,
    ) -> PolarsResult<Vec<Box<dyn Array>>> {
        use polars_core::with_match_physical_numeric_polars_type;

        let ca = self.as_binary();

        match dtype {
            dtype if dtype.is_primitive_numeric() || dtype.is_temporal() => {
                let dtype = dtype.to_physical();
                let arrow_data_type = dtype
                    .to_arrow(CompatLevel::newest())
                    .underlying_physical_type();
                with_match_physical_numeric_polars_type!(dtype, |$T| {
                    unsafe {
                        ca.chunks().iter().map(|chunk| {
                            binview_to_primitive_dyn::<<$T as PolarsNumericType>::Native>(
                                &**chunk,
                                &arrow_data_type,
                                is_little_endian,
                            )
                        }).collect()
                    }
                })
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner_dtype, array_width)
                if inner_dtype.is_primitive_numeric() || inner_dtype.is_temporal() =>
            {
                let inner_dtype = inner_dtype.to_physical();
                let result: Vec<ArrayRef> = with_match_physical_numeric_polars_type!(inner_dtype, |$T| {
                    unsafe {
                        ca.chunks().iter().map(|chunk| {
                            binview_to_fixed_size_list_dyn::<<$T as PolarsNumericType>::Native>(
                                &**chunk,
                                *array_width,
                                is_little_endian
                            )
                        }).collect::<Result<Vec<ArrayRef>, _>>()
                    }
                })?;
                Ok(result)
            },
            _ => Err(
                polars_err!(InvalidOperation: "unsupported data type {:?} in reinterpret. Only numeric or temporal types, or Arrays of those, are allowed.", dtype),
            ),
        }
    }
}

impl BinaryNameSpaceImpl for BinaryChunked {}
