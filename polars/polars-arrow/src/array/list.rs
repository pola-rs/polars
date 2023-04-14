use arrow::array::{new_empty_array, Array, ListArray};
use arrow::bitmap::MutableBitmap;
use arrow::compute::concatenate;
use arrow::datatypes::DataType as ArrowDataType;
use arrow::error::Result;
use arrow::offset::Offsets;

use crate::prelude::*;

pub struct AnonymousBuilder<'a> {
    arrays: Vec<&'a dyn Array>,
    offsets: Vec<i64>,
    validity: Option<MutableBitmap>,
    size: i64,
}

impl<'a> AnonymousBuilder<'a> {
    pub fn new(size: usize) -> Self {
        let mut offsets = Vec::with_capacity(size + 1);
        offsets.push(0i64);
        Self {
            arrays: Vec::with_capacity(size),
            offsets,
            validity: None,
            size: 0,
        }
    }
    #[inline]
    fn last_offset(&self) -> i64 {
        *self.offsets.last().unwrap()
    }

    #[inline]
    fn length(&self) -> usize {
        self.offsets.len() - 1
    }

    #[inline]
    pub fn push(&mut self, arr: &'a dyn Array) {
        self.size += arr.len() as i64;
        self.offsets.push(self.size);
        self.arrays.push(arr);

        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    pub fn push_multiple(&mut self, arrs: &'a [ArrayRef]) {
        for arr in arrs {
            self.size += arr.len() as i64;
            self.arrays.push(arr.as_ref());
        }
        self.offsets.push(self.size);
        self.update_validity()
    }

    pub fn push_null(&mut self) {
        self.offsets.push(self.last_offset());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    pub fn push_empty(&mut self) {
        self.offsets.push(self.last_offset());
        self.update_validity()
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len() - 1;

        let mut validity = MutableBitmap::with_capacity(self.offsets.capacity());
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }

    fn update_validity(&mut self) {
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    pub fn finish(self, inner_dtype: Option<ArrowDataType>) -> Result<ListArray<i64>> {
        let inner_dtype = inner_dtype.unwrap_or_else(|| self.infer_inner_dtype());
        let values = if self.arrays.is_empty() {
            new_empty_array(inner_dtype.clone())
        } else {
            concatenate::concatenate(&self.arrays)?
        };
        // Safety:
        // offsets are monotonically increasing
        let offsets = unsafe { Offsets::new_unchecked(self.offsets).into() };
        Ok(ListArray::<i64>::new(
            ListArray::<i64>::default_datatype(inner_dtype.clone()),
            offsets,
            values,
            self.validity.map(|validity| validity.into()),
        ))
    }

    #[inline]
    fn infer_inner_dtype(&self) -> ArrowDataType {
        if self.arrays.is_empty() {
            if self.length() > 0 {
                ArrowDataType::Int32 // todo NULL_DTYPE
            } else {
                ArrowDataType::Null
            }
        } else {
            self.arrays[0].data_type().clone()
        }
    }
}
