use arrow::array::{Array, BinaryViewArrayGeneric, ViewType};
use crate::parquet::schema::types::PrimitiveType;
use polars_error::PolarsResult;
use crate::parquet::statistics::{BinaryStatistics, ParquetStatistics};
use crate::read::schema::is_nullable;
use crate::write::{Encoding, WriteOptions, Page, utils};
use crate::write::utils::invalid_encoding;

pub fn array_to_page<T: ViewType + ?Sized>(
    array: &BinaryViewArrayGeneric<T>,
    options: WriteOptions,
    type_: PrimitiveType,
    encoding: Encoding
) -> PolarsResult<Page> {
    let is_optional = is_nullable(&type_.field_info);

    let mut buffer = vec![];
    utils::write_def_levels(&mut buffer, is_optional, array.validity(), array.len(), options.version)?;

    let definition_levels_byte_length = buffer.len();

    match encoding {
        Encoding::Plain => todo!(),
        Encoding::DeltaLengthByteArray => todo!(),
        _ => {
            return Err(invalid_encoding(encoding, array.data_type()))
        }
    }

    // let stati
}

// pub(crate) fn build_statistics<T: ViewType + ?Sized>(
//     array: &BinaryViewArrayGeneric<T>,
//     primitive_type: PrimitiveType,
// ) -> ParquetStatistics {
//     let statistics = &BinaryStatistics {
//         primitive_type,
//         null_count: Some(array.null_count() as i64),
//         distinct_count: None,
//         max_value: array
//             .iter()
//             .flatten()
//             .max_by(|x, y| ord_binary(x, y))
//             .map(|x| x.to_vec()),
//         min_value: array
//             .iter()
//             .flatten()
//             .min_by(|x, y| ord_binary(x, y))
//             .map(|x| x.to_vec()),
//     } as &dyn Statistics;
//     serialize_statistics(statistics)
// }
