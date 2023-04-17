use arrow::array::{Array, FixedSizeListArray, ListArray};
use arrow::bitmap::MutableBitmap;
use arrow::compute::concatenate;
use arrow::datatypes::DataType;
use arrow::error::Result;

use crate::prelude::*;

pub struct AnonymousBuilder<'a> {
    arrays: Vec<&'a dyn Array>,
    validity: Option<MutableBitmap>,
    size: usize,
    /// Size of each element of the fixed size list
    inner_size: usize,
}

impl<'a> AnonymousBuilder<'a> {
    pub fn new(size: usize, inner_size: usize) -> Self {
        Self {
            arrays: Vec::with_capacity(size),
            validity: None,
            size: 0,
            inner_size,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    #[inline]
    pub fn push(&mut self, arr: &'a dyn Array) {
        assert_eq!(
            arr.len(),
            self.inner_size,
            "arr has size {} but expected size {}",
            arr.len(),
            self.inner_size
        );

        self.size += arr.len();
        self.arrays.push(arr);

        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    pub fn push_multiple(&mut self, arrs: &'a [ArrayRef]) {
        for arr in arrs {
            self.size += arr.len();
            self.arrays.push(arr.as_ref());
        }
        self.update_validity()
    }

    pub fn push_null(&mut self) {
        // TODO: An Arrow FixedSizeListArray doesn't have offsets, so for a null object we need to
        // still allocate `inner_size` in the array. But we don't necessarily know yet what the
        // inner dtype will be yet.
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    fn init_validity(&mut self) {
        let len = self.size;

        let mut validity = MutableBitmap::with_capacity(self.size);
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }

    fn update_validity(&mut self) {
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    pub fn finish(self, inner_dtype: Option<&DataType>) -> Result<FixedSizeListArray> {
        let inner_dtype = inner_dtype.unwrap_or_else(|| self.arrays[0].data_type());
        let values = concatenate::concatenate(&self.arrays)?;
        let dtype = ListArray::<i64>::default_datatype(inner_dtype.clone());
        Ok(FixedSizeListArray::new(
            dtype,
            values,
            self.validity.map(|validity| validity.into()),
        ))
    }
}
