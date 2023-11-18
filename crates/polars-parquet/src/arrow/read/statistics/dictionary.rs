use arrow::array::*;
use arrow::datatypes::{ArrowDataType, PhysicalType};
use arrow::match_integer_type;
use polars_error::PolarsResult;

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableDictionary {
    data_type: ArrowDataType,
    pub inner: Box<dyn MutableArray>,
}

impl DynMutableDictionary {
    pub fn try_with_capacity(data_type: ArrowDataType, capacity: usize) -> PolarsResult<Self> {
        let inner = if let ArrowDataType::Dictionary(_, inner, _) = &data_type {
            inner.as_ref()
        } else {
            unreachable!()
        };
        let inner = make_mutable(inner, capacity)?;

        Ok(Self { data_type, inner })
    }
}

impl MutableArray for DynMutableDictionary {
    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validity(&self) -> Option<&arrow::bitmap::MutableBitmap> {
        self.inner.validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let inner = self.inner.as_box();
        match self.data_type.to_physical_type() {
            PhysicalType::Dictionary(key) => match_integer_type!(key, |$T| {
                let keys: Vec<$T> = (0..inner.len() as $T).collect();
                let keys = PrimitiveArray::<$T>::from_vec(keys);
                Box::new(DictionaryArray::<$T>::try_new(self.data_type.clone(), keys, inner).unwrap())
            }),
            _ => todo!(),
        }
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
