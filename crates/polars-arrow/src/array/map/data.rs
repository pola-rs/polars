use arrow_data::{ArrayData, ArrayDataBuilder};

use crate::array::{from_data, to_data, Arrow2Arrow, MapArray};
use crate::bitmap::Bitmap;
use crate::offset::OffsetsBuffer;

impl Arrow2Arrow for MapArray {
    fn to_data(&self) -> ArrayData {
        let data_type = self.data_type.clone().into();

        let builder = ArrayDataBuilder::new(data_type)
            .len(self.len())
            .buffers(vec![self.offsets.clone().into_inner().into()])
            .nulls(self.validity.as_ref().map(|b| b.clone().into()))
            .child_data(vec![to_data(self.field.as_ref())]);

        // Safety: Array is valid
        unsafe { builder.build_unchecked() }
    }

    fn from_data(data: &ArrayData) -> Self {
        let data_type = data.data_type().clone().into();
        if data.is_empty() {
            // Handle empty offsets
            return Self::new_empty(data_type);
        }

        let mut offsets = unsafe { OffsetsBuffer::new_unchecked(data.buffers()[0].clone().into()) };
        offsets.slice(data.offset(), data.len() + 1);

        Self {
            data_type: data.data_type().clone().into(),
            offsets,
            field: from_data(&data.child_data()[0]),
            validity: data.nulls().map(|n| Bitmap::from_null_buffer(n.clone())),
        }
    }
}
