use parquet2::indexes::PageIndex;

use crate::{
    array::{Array, BinaryArray, PrimitiveArray, Utf8Array},
    datatypes::{DataType, PhysicalType},
    error::Error,
    trusted_len::TrustedLen,
};

use super::ColumnPageStatistics;

pub fn deserialize(
    indexes: &[PageIndex<Vec<u8>>],
    data_type: &DataType,
) -> Result<ColumnPageStatistics, Error> {
    Ok(ColumnPageStatistics {
        min: deserialize_binary_iter(indexes.iter().map(|index| index.min.as_ref()), data_type)?,
        max: deserialize_binary_iter(indexes.iter().map(|index| index.max.as_ref()), data_type)?,
        null_count: PrimitiveArray::from_trusted_len_iter(
            indexes
                .iter()
                .map(|index| index.null_count.map(|x| x as u64)),
        ),
    })
}

fn deserialize_binary_iter<'a, I: TrustedLen<Item = Option<&'a Vec<u8>>>>(
    iter: I,
    data_type: &DataType,
) -> Result<Box<dyn Array>, Error> {
    match data_type.to_physical_type() {
        PhysicalType::LargeBinary => Ok(Box::new(BinaryArray::<i64>::from_iter(iter))),
        PhysicalType::Utf8 => {
            let iter = iter.map(|x| x.map(|x| std::str::from_utf8(x)).transpose());
            Ok(Box::new(Utf8Array::<i32>::try_from_trusted_len_iter(iter)?))
        }
        PhysicalType::LargeUtf8 => {
            let iter = iter.map(|x| x.map(|x| std::str::from_utf8(x)).transpose());
            Ok(Box::new(Utf8Array::<i64>::try_from_trusted_len_iter(iter)?))
        }
        _ => Ok(Box::new(BinaryArray::<i32>::from_iter(iter))),
    }
}
