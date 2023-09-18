use parquet2::{
    encoding::delta_bitpacked::encode,
    encoding::Encoding,
    page::DataPage,
    schema::types::PrimitiveType,
    statistics::{serialize_statistics, PrimitiveStatistics},
    types::NativeType as ParquetNativeType,
};

use super::super::utils;
use super::super::WriteOptions;
use crate::{
    array::{Array, PrimitiveArray},
    error::Error,
    io::parquet::{read::schema::is_nullable, write::utils::ExactSizedIter},
    types::NativeType,
};

pub(crate) fn encode_plain<T, P>(
    array: &PrimitiveArray<T>,
    is_optional: bool,
    mut buffer: Vec<u8>,
) -> Vec<u8>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
{
    if is_optional {
        buffer.reserve(std::mem::size_of::<P>() * (array.len() - array.null_count()));
        // append the non-null values
        array.iter().for_each(|x| {
            if let Some(x) = x {
                let parquet_native: P = x.as_();
                buffer.extend_from_slice(parquet_native.to_le_bytes().as_ref())
            }
        });
    } else {
        buffer.reserve(std::mem::size_of::<P>() * array.len());
        // append all values
        array.values().iter().for_each(|x| {
            let parquet_native: P = x.as_();
            buffer.extend_from_slice(parquet_native.to_le_bytes().as_ref())
        });
    }
    buffer
}

pub(crate) fn encode_delta<T, P>(
    array: &PrimitiveArray<T>,
    is_optional: bool,
    mut buffer: Vec<u8>,
) -> Vec<u8>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
    P: num_traits::AsPrimitive<i64>,
{
    if is_optional {
        // append the non-null values
        let iterator = array.iter().flatten().map(|x| {
            let parquet_native: P = x.as_();
            let integer: i64 = parquet_native.as_();
            integer
        });
        let iterator = ExactSizedIter::new(iterator, array.len() - array.null_count());
        encode(iterator, &mut buffer)
    } else {
        // append all values
        let iterator = array.values().iter().map(|x| {
            let parquet_native: P = x.as_();
            let integer: i64 = parquet_native.as_();
            integer
        });
        encode(iterator, &mut buffer)
    }
    buffer
}

pub fn array_to_page_plain<T, P>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
) -> Result<DataPage, Error>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
{
    array_to_page(array, options, type_, Encoding::Plain, encode_plain)
}

pub fn array_to_page_integer<T, P>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
) -> Result<DataPage, Error>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
    P: num_traits::AsPrimitive<i64>,
{
    match encoding {
        Encoding::DeltaBinaryPacked => array_to_page(array, options, type_, encoding, encode_delta),
        Encoding::Plain => array_to_page(array, options, type_, encoding, encode_plain),
        other => Err(Error::nyi(format!("Encoding integer as {other:?}"))),
    }
}

pub fn array_to_page<T, P, F: Fn(&PrimitiveArray<T>, bool, Vec<u8>) -> Vec<u8>>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding,
    encode: F,
) -> Result<DataPage, Error>
where
    T: NativeType,
    P: ParquetNativeType,
    // constraint required to build statistics
    T: num_traits::AsPrimitive<P>,
{
    let is_optional = is_nullable(&type_.field_info);

    let validity = array.validity();

    let mut buffer = vec![];
    utils::write_def_levels(
        &mut buffer,
        is_optional,
        validity,
        array.len(),
        options.version,
    )?;

    let definition_levels_byte_length = buffer.len();

    let buffer = encode(array, is_optional, buffer);

    let statistics = if options.write_statistics {
        Some(serialize_statistics(&build_statistics(
            array,
            type_.clone(),
        )))
    } else {
        None
    };

    utils::build_plain_page(
        buffer,
        array.len(),
        array.len(),
        array.null_count(),
        0,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        encoding,
    )
}

pub fn build_statistics<T, P>(
    array: &PrimitiveArray<T>,
    primitive_type: PrimitiveType,
) -> PrimitiveStatistics<P>
where
    T: NativeType,
    P: ParquetNativeType,
    T: num_traits::AsPrimitive<P>,
{
    PrimitiveStatistics::<P> {
        primitive_type,
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array
            .iter()
            .flatten()
            .map(|x| {
                let x: P = x.as_();
                x
            })
            .max_by(|x, y| x.ord(y)),
        min_value: array
            .iter()
            .flatten()
            .map(|x| {
                let x: P = x.as_();
                x
            })
            .min_by(|x, y| x.ord(y)),
    }
}
