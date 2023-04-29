use arrow::array::{Array, FixedSizeListArray};
use arrow::bitmap::{Bitmap, MutableBitmap};
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
            "arr has size {} but expected size {} to match the fixed-size of the list",
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
        let inner_size = self.inner_size;
        let validity = self.validity.map(|validity| validity.into());
        let values = expanding_concatenate(&self.arrays, &validity, inner_dtype, inner_size)?;
        // let values = concatenate::concatenate(&self.arrays)?;
        let dtype = FixedSizeListArray::default_datatype(inner_dtype.clone(), inner_size);
        Ok(FixedSizeListArray::new(dtype, values, validity))
    }
}

fn expanding_concatenate(
    arrays: &[&dyn Array],
    validity: &Option<Bitmap>,
    inner_dtype: &DataType,
    inner_size: usize,
) -> Result<Box<dyn Array>> {
    if arrays.is_empty() {
        return Err(arrow::error::Error::InvalidArgumentError(
            "concat requires input of at least one array".to_string(),
        ));
    }

    if arrays
        .iter()
        .any(|array| array.data_type() != arrays[0].data_type())
    {
        return Err(arrow::error::Error::InvalidArgumentError(
            "It is not possible to concatenate arrays of different data types.".to_string(),
        ));
    }

    if arrays.iter().any(|array| array.len() != inner_size) {
        return Err(arrow::error::Error::InvalidArgumentError(
            "It is not possible to concatenate arrays of different lengths for a fixed size list."
                .to_string(),
        ));
    }

    if let Some(validity) = validity {
        let mut new_arrays: Vec<FixedSizeListArray> = Vec::with_capacity(validity.len());

        let mut counter = 0;
        for valid in validity.into_iter() {
            if valid {
                let array = arrays[counter];
                new_arrays.push(
                    array
                        .as_any()
                        .downcast_ref::<FixedSizeListArray>()
                        .unwrap()
                        .clone(),
                );
                counter += 1;
            } else {
                let null_arr = FixedSizeListArray::new_null(inner_dtype.clone(), inner_size);
                new_arrays.push(null_arr);
            }
        }

        // TODO: this is not great
        let refs: Vec<&dyn Array> = new_arrays.into_iter().map(|arr| &&arr).collect();
        concatenate::concatenate(refs.as_slice())
    } else {
        concatenate::concatenate(arrays)
    }
}
