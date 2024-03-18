use arrow_buffer::{BooleanBuffer, NullBuffer};
use arrow_data::{ArrayData, ArrayDataBuilder};

use crate::array::{Arrow2Arrow, BooleanArray};
use crate::bitmap::Bitmap;
use crate::datatypes::ArrowDataType;

impl Arrow2Arrow for BooleanArray {
    fn to_data(&self) -> ArrayData {
        let buffer = NullBuffer::from(self.values.clone());

        let builder = ArrayDataBuilder::new(arrow_schema::DataType::Boolean)
            .len(buffer.len())
            .offset(buffer.offset())
            .buffers(vec![buffer.into_inner().into_inner()])
            .nulls(self.validity.as_ref().map(|b| b.clone().into()));

        // SAFETY: Array is valid
        unsafe { builder.build_unchecked() }
    }

    fn from_data(data: &ArrayData) -> Self {
        assert_eq!(data.data_type(), &arrow_schema::DataType::Boolean);

        let buffers = data.buffers();
        let buffer = BooleanBuffer::new(buffers[0].clone(), data.offset(), data.len());
        // Use NullBuffer to compute set count
        let values = Bitmap::from_null_buffer(NullBuffer::new(buffer));

        Self {
            data_type: ArrowDataType::Boolean,
            values,
            validity: data.nulls().map(|n| Bitmap::from_null_buffer(n.clone())),
        }
    }
}
