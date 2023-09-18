use crate::{
    array::{Array, MapArray, MutableArray},
    datatypes::DataType,
    error::Error,
};

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableMapArray {
    data_type: DataType,
    pub inner: Box<dyn MutableArray>,
}

impl DynMutableMapArray {
    pub fn try_with_capacity(data_type: DataType, capacity: usize) -> Result<Self, Error> {
        let inner = match data_type.to_logical_type() {
            DataType::Map(inner, _) => inner,
            _ => unreachable!(),
        };
        let inner = make_mutable(inner.data_type(), capacity)?;

        Ok(Self { data_type, inner })
    }
}

impl MutableArray for DynMutableMapArray {
    fn data_type(&self) -> &DataType {
        &self.data_type
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validity(&self) -> Option<&crate::bitmap::MutableBitmap> {
        None
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(MapArray::new(
            self.data_type.clone(),
            vec![0, self.inner.len() as i32].try_into().unwrap(),
            self.inner.as_box(),
            None,
        ))
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
