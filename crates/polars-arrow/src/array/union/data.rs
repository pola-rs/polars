use arrow_data::{ArrayData, ArrayDataBuilder};

use crate::array::{from_data, to_data, Arrow2Arrow, UnionArray};
use crate::buffer::Buffer;
use crate::datatypes::ArrowDataType;

impl Arrow2Arrow for UnionArray {
    fn to_data(&self) -> ArrayData {
        let data_type = arrow_schema::DataType::from(self.data_type.clone());
        let len = self.len();

        let builder = match self.offsets.clone() {
            Some(offsets) => ArrayDataBuilder::new(data_type)
                .len(len)
                .buffers(vec![self.types.clone().into(), offsets.into()])
                .child_data(self.fields.iter().map(|x| to_data(x.as_ref())).collect()),
            None => ArrayDataBuilder::new(data_type)
                .len(len)
                .buffers(vec![self.types.clone().into()])
                .child_data(
                    self.fields
                        .iter()
                        .map(|x| to_data(x.as_ref()).slice(self.offset, len))
                        .collect(),
                ),
        };

        // SAFETY: Array is valid
        unsafe { builder.build_unchecked() }
    }

    fn from_data(data: &ArrayData) -> Self {
        let data_type: ArrowDataType = data.data_type().clone().into();

        let fields = data.child_data().iter().map(from_data).collect();
        let buffers = data.buffers();
        let mut types: Buffer<i8> = buffers[0].clone().into();
        types.slice(data.offset(), data.len());
        let offsets = match buffers.len() == 2 {
            true => {
                let mut offsets: Buffer<i32> = buffers[1].clone().into();
                offsets.slice(data.offset(), data.len());
                Some(offsets)
            },
            false => None,
        };

        // Map from type id to array index
        let map = match &data_type {
            ArrowDataType::Union(_, Some(ids), _) => {
                let mut map = [0; 127];
                for (pos, &id) in ids.iter().enumerate() {
                    map[id as usize] = pos;
                }
                Some(map)
            },
            ArrowDataType::Union(_, None, _) => None,
            _ => unreachable!("must be Union type"),
        };

        Self {
            types,
            map,
            fields,
            offsets,
            data_type,
            offset: data.offset(),
        }
    }
}
