#[cfg(feature = "binary_encoding")]
use std::borrow::Cow;

#[cfg(feature = "binary_encoding")]
use base64::Engine as _;
#[cfg(feature = "binary_encoding")]
use base64::engine::general_purpose;
use memchr::memmem::find;
#[cfg(feature = "binary_encoding")]
use polars_compute::cast::{binview_to_fixed_size_list, binview_to_primitive};
use polars_compute::size::binary_size_bytes;
use polars_core::prelude::arity::{
    broadcast_binary_elementwise_values, unary_elementwise_values, unary_mut_values,
};

use super::*;

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
        unary_mut_values(ca, binary_size_bytes)
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        if strict {
            ca.try_apply_nonnull_values_generic(|s| {
                hex::decode(s).map_err(|_| {
                    polars_err!(
                        ComputeError:
                        "invalid `hex` encoding found; try setting `strict=false` to ignore"
                    )
                })
            })
        } else {
            Ok(ca.apply(|opt_s| opt_s.and_then(|s| hex::decode(s).ok().map(Cow::Owned))))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn hex_encode(&self) -> Series {
        let ca = self.as_binary();
        unsafe {
            ca.apply_values(|s| hex::encode(s).into_bytes().into())
                .cast_unchecked(&DataType::String)
                .unwrap()
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_decode(&self, strict: bool) -> PolarsResult<BinaryChunked> {
        let ca = self.as_binary();
        if strict {
            ca.try_apply_nonnull_values_generic(|s| {
                general_purpose::STANDARD.decode(s).map_err(|_e| {
                    polars_err!(
                        ComputeError:
                        "invalid `base64` encoding found; try setting `strict=false` to ignore"
                    )
                })
            })
        } else {
            Ok(ca.apply(|opt_s| {
                opt_s.and_then(|s| general_purpose::STANDARD.decode(s).ok().map(Cow::Owned))
            }))
        }
    }

    #[cfg(feature = "binary_encoding")]
    fn base64_encode(&self) -> Series {
        let ca = self.as_binary();
        unsafe {
            ca.apply_values(|s| general_purpose::STANDARD.encode(s).into_bytes().into())
                .cast_unchecked(&DataType::String)
                .unwrap()
        }
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
    ) -> PolarsResult<Vec<PlArrayRef>> {
        use polars_core::with_match_physical_numeric_polars_type;

        let ca = self.as_binary();

        match dtype {
            dtype if dtype.is_primitive_numeric() || dtype.is_temporal() => {
                let dtype = dtype.to_physical();
                with_match_physical_numeric_polars_type!(dtype, |$T| {
                    ca.chunks().iter().map(|chunk| {
                        reinterpret_elementwise(&**chunk, |chunk| {
                            Ok(Box::new(binview_to_primitive::<<$T as PolarsNumericType>::Native>(
                                chunk,
                                is_little_endian,
                            )))
                        })
                    }).collect()
                })
            },
            #[cfg(feature = "dtype-array")]
            DataType::Array(inner_dtype, array_width)
                if inner_dtype.is_primitive_numeric() || inner_dtype.is_temporal() =>
            {
                let inner_dtype = inner_dtype.to_physical();
                let result: Vec<PlArrayRef> = with_match_physical_numeric_polars_type!(inner_dtype, |$T| {
                    ca.chunks().iter().map(|chunk| {
                        reinterpret_elementwise(&**chunk, |chunk| {
                            type N = <$T as PolarsNumericType>::Native;
                            let out = if is_little_endian {
                                binview_to_fixed_size_list::<N, true>(chunk, *array_width)
                            } else {
                                binview_to_fixed_size_list::<N, false>(chunk, *array_width)
                            };
                            Ok(Box::new(out?))
                        })
                    }).collect::<Result<Vec<PlArrayRef>, _>>()
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

/// Runs an elementwise `kernel` over `chunk`, reading a scalar chunk's one element once.
#[cfg(feature = "binary_encoding")]
fn reinterpret_elementwise(
    chunk: &dyn PlArray,
    kernel: impl FnOnce(&PlBinaryViewArray) -> PolarsResult<PlArrayRef>,
) -> PolarsResult<PlArrayRef> {
    let length = chunk.len();

    // Sliced down to the one element the chunk repeats, which leaves every buffer holding the
    // single slot it already held, and is therefore `O(1)`.
    let repeated = PlArray::is_scalar(chunk) && length > 1;
    let sliced;
    let operand = if repeated {
        sliced = chunk.sliced(0, 1);
        &*sliced
    } else {
        chunk
    };

    let out = kernel(
        operand
            .as_any()
            .downcast_ref()
            .expect("a chunk of a binary column is a binary view array"),
    )?;

    Ok(if repeated {
        out.new_from_index(0, length)
    } else {
        out
    })
}
