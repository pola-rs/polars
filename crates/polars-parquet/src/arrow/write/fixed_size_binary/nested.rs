use arrow::array::{Array, FixedSizeBinaryArray};
use polars_error::PolarsResult;

use super::encode_plain;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::FixedLenStatistics;
use crate::read::schema::is_nullable;
use crate::write::{EncodeNullability, Encoding, Nested, WriteOptions, nested, utils};

pub fn array_to_page(
    array: &FixedSizeBinaryArray,
    options: WriteOptions,
    type_: PrimitiveType,
    nested: &[Nested],
    statistics: Option<FixedLenStatistics>,
) -> PolarsResult<DataPage> {
    let is_optional = is_nullable(&type_.field_info);
    let encode_options = EncodeNullability::new(is_optional);

    let mut buffer = vec![];
    let (repetition_levels_byte_length, definition_levels_byte_length) =
        nested::write_rep_and_def(options.version, nested, &mut buffer)?;

    encode_plain(array, encode_options, &mut buffer);

    utils::build_plain_page(
        buffer,
        nested::num_values(nested),
        nested[0].len(),
        array.null_count(),
        repetition_levels_byte_length,
        definition_levels_byte_length,
        statistics.map(|x| x.serialize()),
        type_,
        options,
        Encoding::Plain,
    )
}
