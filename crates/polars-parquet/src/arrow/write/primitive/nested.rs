use arrow::array::{Array, PrimitiveArray};
use arrow::types::NativeType as ArrowNativeType;
use polars_error::PolarsResult;

use super::super::{nested, utils, WriteOptions};
use super::basic::{build_statistics, encode_plain};
use crate::arrow::read::schema::is_nullable;
use crate::arrow::write::Nested;
use crate::parquet::encoding::Encoding;
use crate::parquet::page::DataPage;
use crate::parquet::schema::types::PrimitiveType;
use crate::parquet::statistics::serialize_statistics;
use crate::parquet::types::NativeType;

pub fn array_to_page<T, R>(
    array: &PrimitiveArray<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    nested: &[Nested],
) -> PolarsResult<DataPage>
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
