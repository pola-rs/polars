use arrow_data::{ArrayData, ArrayDataBuilder};

use crate::array::{Arrow2Arrow, FixedSizeBinaryArray};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

impl Arrow2Arrow for FixedSizeBinaryArray {
    fn to_data(&self) -> ArrayData {
        let data_type = self.data_type.clone().into();
        let builder = ArrayDataBuilder::new(data_type)
            .len(self.len())
            .buffers(vec![self.values.clone().into()])
            .nulls(self.validity.as_ref().map(|b| b.clone().into()));

        // SAFETY: Array is valid
        unsafe { builder.build_unchecked() }
    }

    fn from_data(data: &ArrayData) -> Self {
        let data_type: ArrowDataType = data.data_type().clone().into();
        let size = match data_type {
            ArrowDataType::FixedSizeBinary(size) => size,
            _ => unreachable!("must be FixedSizeBinary"),
        };

        let mut values: Buffer<u8> = data.buffers()[0].clone().into();
        values.slice(data.offset() * size, data.len() * size);

        Self {
            size,
            data_type,
            values,
            validity: data.nulls().map(|n| Bitmap::from_null_buffer(n.clone())),
        }
    }
}
