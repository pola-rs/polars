use parquet2::schema::types::PrimitiveType;
use parquet2::{encoding::Encoding, page::DataPage};

use super::super::{nested, utils, WriteOptions};
use super::basic::{build_statistics, encode_plain};
use crate::io::parquet::read::schema::is_nullable;
use crate::io::parquet::write::Nested;
use crate::{
    array::{Array, BooleanArray},
    error::Result,
};

pub fn array_to_page(
    array: &BooleanArray,
    options: WriteOptions,
    type_: PrimitiveType,
    nested: &[Nested],
) -> Result<DataPage> {
    let is_optional = is_nullable(&type_.field_info);

    let mut buffer = vec![];
    let (repetition_levels_byte_length, definition_levels_byte_length) =
        nested::write_rep_and_def(options.version, nested, &mut buffer)?;

    encode_plain(array, is_optional, &mut buffer)?;

    let statistics = if options.write_statistics {
        Some(build_statistics(array))
    } else {
        None
    };

    utils::build_plain_page(
        buffer,
        nested::num_values(nested),
        nested[0].len(),
        array.null_count(),
        repetition_levels_byte_length,
        definition_levels_byte_length,
        statistics,
        type_,
        options,
        Encoding::Plain,
    )
}
