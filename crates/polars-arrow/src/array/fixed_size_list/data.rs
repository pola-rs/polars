use arrow_data::{ArrayData, ArrayDataBuilder};

use crate::array::{from_data, to_data, Arrow2Arrow, FixedSizeListArray};
use crate::bitmap::Bitmap;
use crate::datatypes::ArrowDataType;

impl Arrow2Arrow for FixedSizeListArray {
    fn to_data(&self) -> ArrayData {
        let dtype = self.dtype.clone().into();
        let builder = ArrayDataBuilder::new(dtype)
            .len(self.len())
            .nulls(self.validity.as_ref().map(|b| b.clone().into()))
            .child_data(vec![to_data(self.values.as_ref())]);

        // SAFETY: Array is valid
        unsafe { builder.build_unchecked() }
    }

    fn from_data(data: &ArrayData) -> Self {
        let dtype: ArrowDataType = data.data_type().clone().into();
        let length = data.len() - data.offset();
        let size = match dtype {
            ArrowDataType::FixedSizeList(_, size) => size,
            _ => unreachable!("must be FixedSizeList type"),
        };

        let mut values = from_data(&data.child_data()[0]);
        values.slice(data.offset() * size, data.len() * size);

        Self {
            size,
            length,
            dtype,
            values,
            validity: data.nulls().map(|n| Bitmap::from_null_buffer(n.clone())),
        }
    }
}
