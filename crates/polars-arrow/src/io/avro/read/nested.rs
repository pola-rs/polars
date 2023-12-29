use polars_error::{polars_err, PolarsResult};

use crate::array::*;
use crate::bitmap::*;
use crate::datatypes::*;
use crate::offset::{Offset, Offsets};

/// Auxiliary struct
#[derive(Debug)]
pub struct DynMutableListArray<O: Offset> {
    data_type: ArrowDataType,
    offsets: Offsets<O>,
    values: Box<dyn MutableArray>,
    validity: Option<MutableBitmap>,
}

impl<O: Offset> DynMutableListArray<O> {
    pub fn new_from(
        values: Box<dyn MutableArray>,
        data_type: ArrowDataType,
        capacity: usize,
    ) -> Self {
        assert_eq!(values.len(), 0);
        ListArray::<O>::get_child_field(&data_type);
        Self {
            data_type,
            offsets: Offsets::<O>::with_capacity(capacity),
            values,
            validity: None,
        }
    }

    /// The values
    pub fn mut_values(&mut self) -> &mut dyn MutableArray {
        self.values.as_mut()
    }

    #[inline]
    pub fn try_push_valid(&mut self) -> PolarsResult<()> {
        let total_length = self.values.len();
        let offset = self.offsets.last().to_usize();
        let length = total_length
            .checked_sub(offset)
            .ok_or_else(|| polars_err!(ComputeError: "overflow"))?;

        self.offsets.try_push(length)?;
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        Ok(())
    }

    #[inline]
    fn push_null(&mut self) {
        self.offsets.extend_constant(1);
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len_proxy();

        let mut validity = MutableBitmap::new();
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }
}

impl<O: Offset> MutableArray for DynMutableListArray<O> {
    fn len(&self) -> usize {
        self.offsets.len_proxy()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        ListArray::new(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_box(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .boxed()
    }

    fn as_arc(&mut self) -> std::sync::Arc<dyn Array> {
        ListArray::new(
            self.data_type.clone(),
            std::mem::take(&mut self.offsets).into(),
            self.values.as_box(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .arced()
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push_null()
    }

    fn reserve(&mut self, _: usize) {
        todo!();
    }

    fn shrink_to_fit(&mut self) {
        todo!();
    }
}

#[derive(Debug)]
pub struct FixedItemsUtf8Dictionary {
    data_type: ArrowDataType,
    keys: MutablePrimitiveArray<i32>,
    values: Utf8Array<i32>,
}

impl FixedItemsUtf8Dictionary {
    pub fn with_capacity(values: Utf8Array<i32>, capacity: usize) -> Self {
        Self {
            data_type: ArrowDataType::Dictionary(
                IntegerType::Int32,
                Box::new(values.data_type().clone()),
                false,
            ),
            keys: MutablePrimitiveArray::<i32>::with_capacity(capacity),
            values,
        }
    }

    pub fn push_valid(&mut self, key: i32) {
        self.keys.push(Some(key))
    }

    /// pushes a null value
    pub fn push_null(&mut self) {
        self.keys.push(None)
    }
}

impl MutableArray for FixedItemsUtf8Dictionary {
    fn len(&self) -> usize {
        self.keys.len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.keys.validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        Box::new(
            DictionaryArray::try_new(
                self.data_type.clone(),
                std::mem::take(&mut self.keys).into(),
                Box::new(self.values.clone()),
            )
            .unwrap(),
        )
    }

    fn as_arc(&mut self) -> std::sync::Arc<dyn Array> {
        std::sync::Arc::new(
            DictionaryArray::try_new(
                self.data_type.clone(),
                std::mem::take(&mut self.keys).into(),
                Box::new(self.values.clone()),
            )
            .unwrap(),
        )
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push_null()
    }

    fn reserve(&mut self, _: usize) {
        todo!();
    }

    fn shrink_to_fit(&mut self) {
        todo!();
    }
}

/// Auxiliary struct
#[derive(Debug)]
pub struct DynMutableStructArray {
    data_type: ArrowDataType,
    values: Vec<Box<dyn MutableArray>>,
    validity: Option<MutableBitmap>,
}

impl DynMutableStructArray {
    pub fn new(values: Vec<Box<dyn MutableArray>>, data_type: ArrowDataType) -> Self {
        Self {
            data_type,
            values,
            validity: None,
        }
    }

    /// The values
    pub fn mut_values(&mut self, field: usize) -> &mut dyn MutableArray {
        self.values[field].as_mut()
    }

    #[inline]
    pub fn try_push_valid(&mut self) -> PolarsResult<()> {
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
        Ok(())
    }

    #[inline]
    fn push_null(&mut self) {
        self.values.iter_mut().for_each(|x| x.push_null());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    fn init_validity(&mut self) {
        let len = self.len();

        let mut validity = MutableBitmap::new();
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }
}

impl MutableArray for DynMutableStructArray {
    fn len(&self) -> usize {
        self.values[0].len()
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let values = self.values.iter_mut().map(|x| x.as_box()).collect();

        Box::new(StructArray::new(
            self.data_type.clone(),
            values,
            std::mem::take(&mut self.validity).map(|x| x.into()),
        ))
    }

    fn as_arc(&mut self) -> std::sync::Arc<dyn Array> {
        let values = self.values.iter_mut().map(|x| x.as_box()).collect();

        std::sync::Arc::new(StructArray::new(
            self.data_type.clone(),
            values,
            std::mem::take(&mut self.validity).map(|x| x.into()),
        ))
    }

    fn data_type(&self) -> &ArrowDataType {
        &self.data_type
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    #[inline]
    fn push_null(&mut self) {
        self.push_null()
    }

    fn reserve(&mut self, _: usize) {
        todo!();
    }

    fn shrink_to_fit(&mut self) {
        todo!();
    }
}
