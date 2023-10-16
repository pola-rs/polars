use polars_error::PolarsResult;

use crate::array::{new_null_array, Array, ListArray, NullArray, StructArray};
use crate::bitmap::MutableBitmap;
use crate::compute::concatenate;
use crate::datatypes::DataType;
use crate::legacy::kernels::concatenate::concatenate_owned_unchecked;
use crate::legacy::prelude::*;
use crate::offset::Offsets;

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
        self.offsets.len() == 1
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
        self.update_validity()
    }

    #[inline]
    pub fn push_null(&mut self) {
        self.offsets.push(self.last_offset());
        match &mut self.validity {
            Some(validity) => validity.push(false),
            None => self.init_validity(),
        }
    }

    #[inline]
    pub fn push_opt(&mut self, arr: Option<&'a dyn Array>) {
        match arr {
            None => self.push_null(),
            Some(arr) => self.push(arr),
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

    pub fn finish(self, inner_dtype: Option<&DataType>) -> PolarsResult<ListArray<i64>> {
        // Safety:
        // offsets are monotonically increasing
        let offsets = unsafe { Offsets::new_unchecked(self.offsets) };
        let (inner_dtype, values) = if self.arrays.is_empty() {
            let len = *offsets.last() as usize;
            match inner_dtype {
                None => {
                    let values = NullArray::new(DataType::Null, len).boxed();
                    (DataType::Null, values)
                },
                Some(inner_dtype) => {
                    let values = new_null_array(inner_dtype.clone(), len);
                    (inner_dtype.clone(), values)
                },
            }
        } else {
            let inner_dtype = inner_dtype.unwrap_or_else(|| self.arrays[0].data_type());

            // check if there is a dtype that is not `Null`
            // if we find it, we will convert the null arrays
            // to empty arrays of this dtype, otherwise the concat kernel fails.
            let mut non_null_dtype = None;
            if is_nested_null(inner_dtype) {
                for arr in &self.arrays {
                    if !is_nested_null(arr.data_type()) {
                        non_null_dtype = Some(arr.data_type());
                        break;
                    }
                }
            };

            // there are null arrays found, ensure the types are correct.
            if let Some(dtype) = non_null_dtype {
                let arrays = self
                    .arrays
                    .iter()
                    .map(|arr| {
                        if is_nested_null(arr.data_type()) {
                            convert_inner_type(&**arr, dtype)
                        } else {
                            arr.to_boxed()
                        }
                    })
                    .collect::<Vec<_>>();

                let values = concatenate_owned_unchecked(&arrays)?;
                (dtype.clone(), values)
            } else {
                let values = concatenate::concatenate(&self.arrays)?;
                (inner_dtype.clone(), values)
            }
        };
        let dtype = ListArray::<i64>::default_datatype(inner_dtype);
        Ok(ListArray::<i64>::new(
            dtype,
            offsets.into(),
            values,
            self.validity.map(|validity| validity.into()),
        ))
    }
}

fn is_nested_null(data_type: &DataType) -> bool {
    match data_type {
        DataType::Null => true,
        DataType::LargeList(field) => is_nested_null(field.data_type()),
        DataType::Struct(fields) => fields.iter().all(|field| is_nested_null(field.data_type())),
        _ => false,
    }
}

/// Cast null arrays to inner type and ensure that all offsets remain correct
pub fn convert_inner_type(array: &dyn Array, dtype: &DataType) -> Box<dyn Array> {
    match dtype {
        DataType::LargeList(field) => {
            let array = array.as_any().downcast_ref::<LargeListArray>().unwrap();
            let inner = array.values();
            let new_values = convert_inner_type(inner.as_ref(), field.data_type());
            let dtype = LargeListArray::default_datatype(new_values.data_type().clone());
            LargeListArray::new(
                dtype,
                array.offsets().clone(),
                new_values,
                array.validity().cloned(),
            )
            .boxed()
        },
        DataType::Struct(fields) => {
            let array = array.as_any().downcast_ref::<StructArray>().unwrap();
            let inner = array.values();
            let new_values = inner
                .iter()
                .zip(fields)
                .map(|(arr, field)| convert_inner_type(arr.as_ref(), field.data_type()))
                .collect::<Vec<_>>();
            StructArray::new(dtype.clone(), new_values, array.validity().cloned()).boxed()
        },
        _ => new_null_array(dtype.clone(), array.len()),
    }
}
