#[cfg(feature = "binary_encoding")]
use std::borrow::Cow;

#[cfg(feature = "binary_encoding")]
use arrow::array::Array;
use arrow::array::FixedSizeListArray;
#[cfg(feature = "binary_encoding")]
use arrow::datatypes::PhysicalType;
use arrow::with_match_primitive_type;
#[cfg(feature = "binary_encoding")]
use base64::Engine as _;
#[cfg(feature = "binary_encoding")]
use base64::engine::general_purpose;
use memchr::memmem::find;
use polars_compute::size::binary_size_bytes;
use polars_core::prelude::arity::{broadcast_binary_elementwise_values, unary_elementwise_values};

use super::cast_binary_to_numerical::{
    cast_binview_to_array_primitive_dyn, cast_binview_to_primitive_dyn,
};
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
    #[allow(clippy::wrong_self_convention)]
    fn from_buffer(&self, dtype: &DataType, is_little_endian: bool) -> PolarsResult<Series> {
        unsafe {
            Ok(Series::from_chunks_and_dtype_unchecked(
                self.as_binary().name().clone(),
                self._from_buffer_inner(dtype, is_little_endian)?,
                dtype,
            ))
        }
    }

    fn _from_buffer_inner(
        &self,
        dtype: &DataType,
        is_little_endian: bool,
    ) -> PolarsResult<Vec<Box<dyn Array>>> {
        // We do the conversion for Arrays using a linear version of the dtype,
        // and then later reshape into the correct size.
        let linear_dtype = if let Some(ref shape) = dtype.get_shape() {
            &DataType::Array(Box::new(dtype.leaf_dtype().clone()), shape.iter().product())
        } else {
            dtype
        };
        let arrow_data_type = linear_dtype.to_arrow(CompatLevel::newest());
        let ca = self.as_binary();

        match arrow_data_type.to_physical_type() {
            PhysicalType::Primitive(ty) => {
                with_match_primitive_type!(ty, |$T| {
                    unsafe {
                        ca.chunks().iter().map(|chunk| {
                            cast_binview_to_primitive_dyn::<$T>(
                                &**chunk,
                                &arrow_data_type,
                                is_little_endian,
                            )
                        }).collect()
                    }
                })
            },
            PhysicalType::FixedSizeList => {
                let leaf_dtype = dtype.leaf_dtype();
                let leaf_physical_type = leaf_dtype
                    .to_arrow(CompatLevel::newest())
                    .to_physical_type();
                let primitive_type = if let PhysicalType::Primitive(x) = leaf_physical_type {
                    x
                } else {
                    return Err(
                        polars_err!(InvalidOperation:"unsupported data type in from_buffer. Only numerical types are allowed in arrays."),
                    );
                };
                // Since we know it's a physical size, we
                let element_size = leaf_dtype.byte_size().unwrap();

                let result: Vec<ArrayRef> = with_match_primitive_type!(primitive_type, |$T| {
                    unsafe {
                        ca.chunks().iter().map(|chunk| {
                            cast_binview_to_array_primitive_dyn::<$T>(
                                &**chunk,
                                &arrow_data_type,
                                is_little_endian,
                                element_size
                            )
                        }).collect::<Result<Vec<ArrayRef>, _>>()
                    }
                })?;
                if let Some(shape) = dtype.get_shape() {
                    let mut dimensions: Vec<ReshapeDimension> = shape
                        .iter()
                        .map(|&v| ReshapeDimension::new(v as i64))
                        .collect();
                    dimensions.insert(0, ReshapeDimension::Infer);
                    result
                        .into_iter()
                        .map(|arr| {
                            let fixed_arr = arr
                                .as_ref()
                                .as_any()
                                .downcast_ref::<FixedSizeListArray>()
                                .unwrap();
                            FixedSizeListArray::from_shape(fixed_arr.values().clone(), &dimensions)
                                .map(|arr| arr.with_validity(fixed_arr.validity().cloned()))
                        })
                        .collect::<Result<Vec<ArrayRef>, _>>()
                } else {
                    Ok(result)
                }
            },
            _ => Err(
                polars_err!(InvalidOperation:"unsupported data type in from_buffer. Only numerical types are allowed."),
            ),
        }
    }
}

impl BinaryNameSpaceImpl for BinaryChunked {}
