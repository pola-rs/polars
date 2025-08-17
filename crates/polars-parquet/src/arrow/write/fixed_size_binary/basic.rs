use arrow::array::{Array, FixedSizeBinaryArray};
use polars_error::PolarsResult;

use super::encode_plain;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::FixedLenStatistics;
use crate::read::schema::is_nullable;
use crate::write::{EncodeNullability, Encoding, WriteOptions, utils};

pub fn array_to_page(
    array: &FixedSizeBinaryArray,
    options: WriteOptions,
    type_: PrimitiveType,
    statistics: Option<FixedLenStatistics>,
) -> PolarsResult<DataPage> {
    let is_optional = is_nullable(&type_.field_info);
    let encode_options = EncodeNullability::new(is_optional);

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

    encode_plain(array, encode_options, &mut buffer);

    utils::build_plain_page(
        buffer,
        array.len(),
        array.len(),
        array.null_count(),
        0,
        definition_levels_byte_length,
        statistics.map(|x| x.serialize()),
        type_,
        options,
        Encoding::Plain,
    )
}
