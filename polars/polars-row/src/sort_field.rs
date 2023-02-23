use super::*;
use encodings::fixed::FixedLengthEncoding;
use arrow::datatypes::DataType as ArrowDataType;

pub struct SortField {
    /// Whether to sort in descending order
    pub(crate) descending: bool,
    /// Whether to sort nulls first
    pub(crate) nulls_first: bool,
    /// Data type
    pub(crate) data_type: ArrowDataType,
}

impl SortField {
    pub(crate) fn encoded_size(&self) -> usize {
        use ArrowDataType::*;
        match self.data_type {
            UInt8 => u8::ENCODED_LEN,
            UInt16 => u16::ENCODED_LEN,
            UInt32 => u32::ENCODED_LEN,
            UInt64 => u64::ENCODED_LEN,
            Int8 => i8::ENCODED_LEN,
            Int16 => i16::ENCODED_LEN,
            Int32 => i32::ENCODED_LEN,
            Int64 => i64::ENCODED_LEN,
            Float32 => f32::ENCODED_LEN,
            Float64 => f64::ENCODED_LEN,
            _ => unimplemented!()
        }
    }
}

