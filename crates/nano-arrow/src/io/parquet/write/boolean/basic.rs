use parquet2::{
    encoding::{hybrid_rle::bitpacked_encode, Encoding},
    page::DataPage,
    schema::types::PrimitiveType,
    statistics::{serialize_statistics, BooleanStatistics, ParquetStatistics, Statistics},
};

use super::super::utils;
use super::super::WriteOptions;
use crate::array::*;
use crate::{error::Result, io::parquet::read::schema::is_nullable};

fn encode(iterator: impl Iterator<Item = bool>, buffer: &mut Vec<u8>) -> Result<()> {
    // encode values using bitpacking
    let len = buffer.len();
    let mut buffer = std::io::Cursor::new(buffer);
    buffer.set_position(len as u64);
    Ok(bitpacked_encode(&mut buffer, iterator)?)
}

pub(super) fn encode_plain(
    array: &BooleanArray,
    is_optional: bool,
    buffer: &mut Vec<u8>,
) -> Result<()> {
    if is_optional {
        let iter = array.iter().flatten().take(
            array
                .validity()
                .as_ref()
                .map(|x| x.len() - x.unset_bits())
                .unwrap_or_else(|| array.len()),
        );
        encode(iter, buffer)
    } else {
        let iter = array.values().iter();
        encode(iter, buffer)
    }
}

pub fn array_to_page(
    array: &BooleanArray,
    options: WriteOptions,
    type_: PrimitiveType,
) -> Result<DataPage> {
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

    encode_plain(array, is_optional, &mut buffer)?;

    let statistics = if options.write_statistics {
        Some(build_statistics(array))
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
        Encoding::Plain,
    )
}

pub(super) fn build_statistics(array: &BooleanArray) -> ParquetStatistics {
    let statistics = &BooleanStatistics {
        null_count: Some(array.null_count() as i64),
        distinct_count: None,
        max_value: array.iter().flatten().max(),
        min_value: array.iter().flatten().min(),
    } as &dyn Statistics;
    serialize_statistics(statistics)
}
