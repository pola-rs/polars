#[cfg(feature = "binary_encoding")]
use std::borrow::Cow;

#[cfg(feature = "binary_encoding")]
use arrow::array::Array;
#[cfg(feature = "binary_encoding")]
use base64::Engine as _;
#[cfg(feature = "binary_encoding")]
use base64::engine::general_purpose;
use memchr::memmem::find;
use polars_compute::cast::{binview_to_fixed_size_list_dyn, binview_to_primitive_dyn};
use polars_compute::size::binary_size_bytes;
use polars_core::prelude::arity::{broadcast_binary_elementwise_values, unary_elementwise_values};

use super::*;

pub trait BinaryNameSpaceImpl: AsBinary {
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
