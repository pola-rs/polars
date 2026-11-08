use polars_arrow::array::BooleanArray;
use polars_error::PolarsResult;

use super::super::{EncodeNullability, WriteOptions, nested, utils};
use super::basic::{build_statistics, encode_plain};
use crate::arrow::read::schema::is_nullable;
use crate::arrow::write::Nested;
use crate::parquet::encoding::Encoding;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;

pub fn array_to_page(
    array: &BooleanArray,
    options: WriteOptions,
    type_: PrimitiveType,
    nested: &[Nested],
) -> PolarsResult<DataPage> {
    let is_optional = is_nullable(&type_.field_info);
    let encode_options = EncodeNullability::new(is_optional);

    let mut buffer = vec![];
    let (repetition_levels_byte_length, definition_levels_byte_length, num_values) =
        nested::write_rep_and_def(options.version, nested, &mut buffer)?;

    encode_plain(array, encode_options, &mut buffer)?;

    let statistics = if options.has_statistics() {
        Some(build_statistics(array, &options.statistics))
    } else {
        None
    };
    utils::build_plain_page(
        buffer,
        num_values,
        nested[0].len(),
        nested::num_nulls(num_values, array),
        repetition_levels_byte_length,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        Encoding::Plain,
    )
}
