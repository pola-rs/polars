mod basic;
pub(super) mod decoders;
pub(super) mod utils;

use arrow::array::{Array, BinaryArray, Utf8Array};
use arrow::bitmap::MutableBitmap;
use arrow::datatypes::{ArrowDataType, PhysicalType};
pub(crate) use basic::BinaryDecoder;

use self::utils::Binary;
use super::utils::freeze_validity;
use super::ParquetResult;

fn finalize(
    data_type: ArrowDataType,
    mut values: Binary,
    validity: MutableBitmap,
) -> ParquetResult<Box<dyn Array>> {
    values.offsets.shrink_to_fit();
    values.values.shrink_to_fit();
    let validity = freeze_validity(validity);

    match data_type.to_physical_type() {
        PhysicalType::LargeBinary => unsafe {
            Ok(BinaryArray::new_unchecked(
                data_type,
                values.offsets.into(),
                values.values.into(),
                validity,
            )
            .boxed())
        },
        PhysicalType::LargeUtf8 => unsafe {
            Ok(Utf8Array::new_unchecked(
                data_type,
                values.offsets.into(),
                values.values.into(),
                validity,
            )
            .boxed())
        },
        _ => unreachable!(),
    }
}
