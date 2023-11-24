use polars_error::PolarsResult;

use crate::array::{ArrayRef, FixedSizeListArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;
use crate::legacy::kernels::concatenate::concatenate_owned_unchecked;

#[derive(Default)]
pub struct AnonymousBuilder {
    arrays: Vec<ArrayRef>,
    validity: Option<MutableBitmap>,
    pub width: usize,
}

impl AnonymousBuilder {
    pub fn new(capacity: usize, width: usize) -> Self {
        Self {
            arrays: Vec::with_capacity(capacity),
            validity: None,
            width,
        }
    }
    pub fn is_empty(&self) -> bool {
        self.arrays.is_empty()
    }

    #[inline]
    pub fn push(&mut self, arr: ArrayRef) {
        self.arrays.push(arr);

        if let Some(validity) = &mut self.validity {
            validity.push(true)
        }
    }

    pub fn push_null(&mut self) {
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.arrays.capacity());
        validity.extend_constant(self.arrays.len(), true);
        validity.set(self.arrays.len() - 1, false);
        self.validity = Some(validity)
    }

    pub fn finish(self, inner_dtype: Option<&ArrowDataType>) -> PolarsResult<FixedSizeListArray> {
        let values = concatenate_owned_unchecked(&self.arrays)?;
        let inner_dtype = inner_dtype.unwrap_or_else(|| self.arrays[0].data_type());
        let data_type = FixedSizeListArray::default_datatype(inner_dtype.clone(), self.width);
        Ok(FixedSizeListArray::new(
            data_type,
            values,
            self.validity.map(|validity| validity.into()),
        ))
    }
}
