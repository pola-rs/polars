use std::sync::Arc;

use polars_core::error::PolarsResult;
use polars_core::prelude::{Column, DataType, IntoColumn};
use polars_ops::prelude::BinaryNameSpaceImpl;
use polars_plan::dsl::{ColumnsUdf, SpecialEq};
use polars_plan::plans::IRBinaryFunction;

pub fn function_expr_to_udf(func: IRBinaryFunction) -> SpecialEq<Arc<dyn ColumnsUdf>> {
    use IRBinaryFunction::*;
    match func {
        Contains => {
            map_as_slice!(contains)
        },
        EndsWith => {
            map_as_slice!(ends_with)
        },
        StartsWith => {
            map_as_slice!(starts_with)
        },
        #[cfg(feature = "binary_encoding")]
        HexDecode(strict) => map!(hex_decode, strict),
        #[cfg(feature = "binary_encoding")]
        HexEncode => map!(hex_encode),
        #[cfg(feature = "binary_encoding")]
        Base64Decode(strict) => map!(base64_decode, strict),
        #[cfg(feature = "binary_encoding")]
        Base64Encode => map!(base64_encode),
        Size => map!(size_bytes),
        #[cfg(feature = "binary_encoding")]
        Reinterpret(dtype, is_little_endian) => map!(reinterpret, &dtype, is_little_endian),
        Slice => {
            map_as_slice!(bin_slice)
        },
        Head => {
            map_as_slice!(bin_head)
        },
        Tail => {
            map_as_slice!(bin_tail)
        },
        Get(null_on_oob) => {
            map_as_slice!(bin_get, null_on_oob)
        },
    }
}

pub(super) fn contains(s: &[Column]) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    let lit = s[1].binary()?;
    Ok(ca
        .contains_chunked(lit)?
        .with_name(ca.name().clone())
        .into_column())
}

pub(super) fn ends_with(s: &[Column]) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    let suffix = s[1].binary()?;

    Ok(ca
        .ends_with_chunked(suffix)?
        .with_name(ca.name().clone())
        .into_column())
}

pub(super) fn starts_with(s: &[Column]) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    let prefix = s[1].binary()?;

    Ok(ca
        .starts_with_chunked(prefix)?
        .with_name(ca.name().clone())
        .into_column())
}

pub(super) fn size_bytes(s: &Column) -> PolarsResult<Column> {
    let ca = s.binary()?;
    Ok(ca.size_bytes().into_column())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn hex_decode(s: &Column, strict: bool) -> PolarsResult<Column> {
    let ca = s.binary()?;
    ca.hex_decode(strict).map(|ok| ok.into_column())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn hex_encode(s: &Column) -> PolarsResult<Column> {
    let ca = s.binary()?;
    Ok(ca.hex_encode().into())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn base64_decode(s: &Column, strict: bool) -> PolarsResult<Column> {
    let ca = s.binary()?;
    ca.base64_decode(strict).map(|ok| ok.into_column())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn base64_encode(s: &Column) -> PolarsResult<Column> {
    let ca = s.binary()?;
    Ok(ca.base64_encode().into())
}

#[cfg(feature = "binary_encoding")]
pub(super) fn reinterpret(
    s: &Column,
    dtype: &DataType,
    is_little_endian: bool,
) -> PolarsResult<Column> {
    let ca = s.binary()?;
    ca.reinterpret(dtype, is_little_endian)
        .map(|val| val.into())
}

pub(super) fn bin_slice(s: &mut [Column]) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    Ok(ca
        .bin_slice(&s[1], &s[2])?
        .with_name(ca.name().clone())
        .into_column())
}

pub(super) fn bin_head(s: &mut [Column]) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    Ok(ca
        .bin_head(&s[1])?
        .with_name(ca.name().clone())
        .into_column())
}

pub(super) fn bin_tail(s: &mut [Column]) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    Ok(ca
        .bin_tail(&s[1])?
        .with_name(ca.name().clone())
        .into_column())
}

pub(super) fn bin_get(s: &mut [Column], null_on_oob: bool) -> PolarsResult<Column> {
    let ca = s[0].binary()?;
    let index = s[1].cast(&DataType::Int64)?;
    polars_ops::prelude::bin_get(ca, index.i64()?, null_on_oob)
}
