mod binary;
mod fixed_len_binary;
mod primitive;

pub use binary::BinaryPageDict;
pub use fixed_len_binary::FixedLenByteArrayPageDict;
use polars_parquet::parquet::error::{ParquetError, ParquetResult};
use polars_parquet::parquet::page::DictPage;
use polars_parquet::parquet::schema::types::PhysicalType;
pub use primitive::PrimitivePageDict;

pub enum DecodedDictPage {
    Int32(PrimitivePageDict<i32>),
    Int64(PrimitivePageDict<i64>),
    Int96(PrimitivePageDict<[u32; 3]>),
    Float(PrimitivePageDict<f32>),
    Double(PrimitivePageDict<f64>),
    ByteArray(BinaryPageDict),
    FixedLenByteArray(FixedLenByteArrayPageDict),
}

pub fn deserialize(page: &DictPage, physical_type: PhysicalType) -> ParquetResult<DecodedDictPage> {
    _deserialize(&page.buffer, page.num_values, page.is_sorted, physical_type)
}

fn _deserialize(
    buf: &[u8],
    num_values: usize,
    is_sorted: bool,
    physical_type: PhysicalType,
) -> ParquetResult<DecodedDictPage> {
    match physical_type {
        PhysicalType::Boolean => Err(ParquetError::OutOfSpec(
            "Boolean physical type cannot be dictionary-encoded".to_string(),
        )),
        PhysicalType::Int32 => {
            primitive::read::<i32>(buf, num_values, is_sorted).map(DecodedDictPage::Int32)
        },
        PhysicalType::Int64 => {
            primitive::read::<i64>(buf, num_values, is_sorted).map(DecodedDictPage::Int64)
        },
        PhysicalType::Int96 => {
            primitive::read::<[u32; 3]>(buf, num_values, is_sorted).map(DecodedDictPage::Int96)
        },
        PhysicalType::Float => {
            primitive::read::<f32>(buf, num_values, is_sorted).map(DecodedDictPage::Float)
        },
        PhysicalType::Double => {
            primitive::read::<f64>(buf, num_values, is_sorted).map(DecodedDictPage::Double)
        },
        PhysicalType::ByteArray => binary::read(buf, num_values).map(DecodedDictPage::ByteArray),
        PhysicalType::FixedLenByteArray(size) => {
            fixed_len_binary::read(buf, size, num_values).map(DecodedDictPage::FixedLenByteArray)
        },
    }
}
