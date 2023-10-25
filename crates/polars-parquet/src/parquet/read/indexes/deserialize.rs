use parquet_format_safe::{thrift::protocol::TCompactInputProtocol, ColumnIndex};

use crate::error::Error;
use crate::schema::types::{PhysicalType, PrimitiveType};

use crate::indexes::{BooleanIndex, ByteIndex, FixedLenByteIndex, Index, NativeIndex};

pub fn deserialize(data: &[u8], primitive_type: PrimitiveType) -> Result<Box<dyn Index>, Error> {
    let mut prot = TCompactInputProtocol::new(data, data.len() * 2 + 1024);

    let index = ColumnIndex::read_from_in_protocol(&mut prot)?;

    let index = match primitive_type.physical_type {
        PhysicalType::Boolean => Box::new(BooleanIndex::try_new(index)?) as Box<dyn Index>,
        PhysicalType::Int32 => Box::new(NativeIndex::<i32>::try_new(index, primitive_type)?),
        PhysicalType::Int64 => Box::new(NativeIndex::<i64>::try_new(index, primitive_type)?),
        PhysicalType::Int96 => Box::new(NativeIndex::<[u32; 3]>::try_new(index, primitive_type)?),
        PhysicalType::Float => Box::new(NativeIndex::<f32>::try_new(index, primitive_type)?),
        PhysicalType::Double => Box::new(NativeIndex::<f64>::try_new(index, primitive_type)?),
        PhysicalType::ByteArray => Box::new(ByteIndex::try_new(index, primitive_type)?),
        PhysicalType::FixedLenByteArray(_) => {
            Box::new(FixedLenByteIndex::try_new(index, primitive_type)?)
        }
    };

    Ok(index)
}
