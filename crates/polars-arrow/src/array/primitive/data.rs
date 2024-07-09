use arrow_data::{ArrayData, ArrayDataBuilder};

use crate::array::{Arrow2Arrow, PrimitiveArray};
use crate::bitmap::Bitmap;
use crate::buffer::Buffer;
use crate::types::NativeType;

impl<T: NativeType> Arrow2Arrow for PrimitiveArray<T> {
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
        let data_type = data.data_type().clone().into();

        let mut values: Buffer<T> = data.buffers()[0].clone().into();
        values.slice(data.offset(), data.len());

        Self {
            data_type,
            values,
            validity: data.nulls().map(|n| Bitmap::from_null_buffer(n.clone())),
        }
    }
}
