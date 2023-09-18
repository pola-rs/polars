use crate::array::{Array, StructArray};
use crate::error::Result;
use crate::{array::MutableArray, datatypes::DataType};

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableStructArray {
    data_type: DataType,
    pub inner: Vec<Box<dyn MutableArray>>,
}

impl DynMutableStructArray {
    pub fn try_with_capacity(data_type: DataType, capacity: usize) -> Result<Self> {
        let inners = match data_type.to_logical_type() {
            DataType::Struct(inner) => inner,
            _ => unreachable!(),
        };
        let inner = inners
            .iter()
            .map(|f| make_mutable(f.data_type(), capacity))
            .collect::<Result<Vec<_>>>()?;

        Ok(Self { data_type, inner })
    }
}
impl MutableArray for DynMutableStructArray {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn len(&self) -> usize {
        self.inner[0].len()
    }

    fn validity(&self) -> Option<&crate::bitmap::MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let inner = self.inner.iter_mut().map(|x| x.as_box()).collect();

        Box::new(StructArray::new(self.data_type.clone(), inner, None))
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn push_null(&mut self) {
        todo!()
    }

    fn reserve(&mut self, _: usize) {
        todo!();
    }

    fn shrink_to_fit(&mut self) {
        todo!()
    }
}
