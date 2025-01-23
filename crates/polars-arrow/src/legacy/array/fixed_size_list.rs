use polars_error::PolarsResult;

use crate::array::{new_null_array, ArrayRef, FixedSizeListArray, NullArray};
use crate::bitmap::BitmapBuilder;
use crate::datatypes::ArrowDataType;
use crate::legacy::array::{convert_inner_type, is_nested_null};
use crate::legacy::kernels::concatenate::concatenate_owned_unchecked;

#[derive(Default)]
pub struct AnonymousBuilder {
    arrays: Vec<ArrayRef>,
    validity: Option<BitmapBuilder>,
    length: usize,
    pub width: usize,
}

impl AnonymousBuilder {
    pub fn new(capacity: usize, width: usize) -> Self {
        Self {
            arrays: Vec::with_capacity(capacity),
            validity: None,
            width,
            length: 0,
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

        self.length += 1;
    }

    pub fn push_null(&mut self) {
        self.arrays
            .push(NullArray::new(ArrowDataType::Null, self.width).boxed());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }

        self.length += 1;
    }

    fn init_validity(&mut self) {
        let mut validity = BitmapBuilder::with_capacity(self.arrays.capacity());
        if !self.arrays.is_empty() {
            validity.extend_constant(self.arrays.len() - 1, true);
            validity.push(false);
        }
        self.validity = Some(validity)
    }

    pub fn finish(self, inner_dtype: Option<&ArrowDataType>) -> PolarsResult<FixedSizeListArray> {
        let mut inner_dtype = inner_dtype.unwrap_or_else(|| self.arrays[0].dtype());

        if is_nested_null(inner_dtype) {
            for arr in &self.arrays {
                if !is_nested_null(arr.dtype()) {
                    inner_dtype = arr.dtype();
                    break;
                }
            }
        };

        // convert nested null arrays to the correct dtype.
        let arrays = self
            .arrays
            .iter()
            .map(|arr| {
                if matches!(arr.dtype(), ArrowDataType::Null) {
                    new_null_array(inner_dtype.clone(), arr.len())
                } else if is_nested_null(arr.dtype()) {
                    convert_inner_type(&**arr, inner_dtype)
                } else {
                    arr.to_boxed()
                }
            })
            .collect::<Vec<_>>();

        let values = concatenate_owned_unchecked(&arrays)?;

        let dtype = FixedSizeListArray::default_datatype(inner_dtype.clone(), self.width);
        Ok(FixedSizeListArray::new(
            dtype,
            self.length,
            values,
            self.validity
                .and_then(|validity| validity.into_opt_validity()),
        ))
    }
}
