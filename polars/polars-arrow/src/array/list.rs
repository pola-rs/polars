use arrow::array::{Array, ListArray};
use arrow::bitmap::MutableBitmap;
use arrow::compute::concatenate;
use arrow::datatypes::DataType;
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

    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    pub fn offsets(&self) -> &[i64] {
        &self.offsets
    }

    pub fn take_offsets(self) -> Offsets<i64> {
        // safety: offsets are correct
        unsafe { Offsets::new_unchecked(self.offsets) }
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
        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
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
    }

    fn init_validity(&mut self) {
        let len = self.offsets.len() - 1;

        let mut validity = MutableBitmap::with_capacity(self.offsets.capacity());
        validity.extend_constant(len, true);
        validity.set(len - 1, false);
        self.validity = Some(validity)
    }

    pub fn finish(self, inner_dtype: Option<&DataType>) -> Result<ListArray<i64>> {
        let inner_dtype = inner_dtype.unwrap_or_else(|| self.arrays[0].data_type());
        let values = concatenate::concatenate(&self.arrays)?;

        let dtype = ListArray::<i64>::default_datatype(inner_dtype.clone());
        // Safety:
        // offsets are monotonically increasing
        unsafe {
            Ok(ListArray::<i64>::new(
                dtype,
                Offsets::new_unchecked(self.offsets).into(),
                values,
                self.validity.map(|validity| validity.into()),
            ))
        }
    }
}
