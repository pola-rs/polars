use parquet2::schema::types::PrimitiveType;
use parquet2::statistics::serialize_statistics;
use parquet2::{encoding::Encoding, page::DataPage, types::NativeType};

use super::super::nested;
use super::super::utils;
use super::super::WriteOptions;
use super::basic::{build_statistics, encode_plain};
use crate::io::parquet::read::schema::is_nullable;
use crate::io::parquet::write::Nested;
use crate::{
    array::{Array, PrimitiveArray},
    error::Result,
    types::NativeType as ArrowNativeType,
};

pub fn array_to_page<T, R>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    nested: &[Nested],
) -> Result<DataPage>
where
    T: ArrowNativeType,
    R: NativeType,
    T: num_traits::AsPrimitive<R>,
{
    let is_optional = is_nullable(&type_.field_info);

    let mut buffer = vec![];

    let (repetition_levels_byte_length, definition_levels_byte_length) =
        nested::write_rep_and_def(options.version, nested, &mut buffer)?;

    let buffer = encode_plain(array, is_optional, buffer);

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
