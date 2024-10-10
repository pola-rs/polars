use std::sync::Arc;

use polars_error::{polars_bail, PolarsResult};

use super::StructArray;
use crate::array::{Array, MutableArray};
use crate::bitmap::MutableBitmap;
use crate::datatypes::ArrowDataType;

/// Converting a [`MutableStructArray`] into a [`StructArray`] is `O(1)`.
#[derive(Debug)]
pub struct MutableStructArray {
    dtype: ArrowDataType,
    length: usize,
    values: Vec<Box<dyn MutableArray>>,
    validity: Option<MutableBitmap>,
}

fn check(
    dtype: &ArrowDataType,
    length: usize,
    values: &[Box<dyn MutableArray>],
    validity: Option<usize>,
) -> PolarsResult<()> {
    let fields = StructArray::try_get_fields(dtype)?;

    if fields.len() != values.len() {
        polars_bail!(ComputeError: "a StructArray must have a number of fields in its DataType equal to the number of child values")
    }

    fields
            .iter().map(|a| &a.dtype)
            .zip(values.iter().map(|a| a.dtype()))
            .enumerate()
            .try_for_each(|(index, (dtype, child))| {
                if dtype != child {
                    polars_bail!(ComputeError: "The children DataTypes of a StructArray must equal the children data types.\nHowever, the field {index} has data type {dtype:?} but the value has data type {child:?}")
                } else {
                    Ok(())
                }
            })?;

    values
            .iter()
            .map(|f| f.len())
            .enumerate()
            .try_for_each(|(index, f_length)| {
                if f_length != length {
                    polars_bail!(ComputeError: "The children must have the given number of values.\nHowever, the values at index {index} have a length of {f_length}, which is different from given length {length}.")
                } else {
                    Ok(())
                }
            })?;

    if validity.map_or(false, |validity| validity != length) {
        polars_bail!(ComputeError:
            "the validity length of a StructArray must match its number of elements",
        )
    }
    Ok(())
}

impl From<MutableStructArray> for StructArray {
    fn from(other: MutableStructArray) -> Self {
        let validity = if other.validity.as_ref().map(|x| x.unset_bits()).unwrap_or(0) > 0 {
            other.validity.map(|x| x.into())
        } else {
            None
        };

        StructArray::new(
            other.dtype,
            other.length,
            other.values.into_iter().map(|mut v| v.as_box()).collect(),
            validity,
        )
    }
}

impl MutableStructArray {
    /// Creates a new [`MutableStructArray`].
    pub fn new(dtype: ArrowDataType, length: usize, values: Vec<Box<dyn MutableArray>>) -> Self {
        Self::try_new(dtype, length, values, None).unwrap()
    }

    /// Create a [`MutableStructArray`] out of low-end APIs.
    /// # Errors
    /// This function errors iff:
    /// * `dtype` is not [`ArrowDataType::Struct`]
    /// * The inner types of `dtype` are not equal to those of `values`
    /// * `validity` is not `None` and its length is different from the `values`'s length
    pub fn try_new(
        dtype: ArrowDataType,
        length: usize,
        values: Vec<Box<dyn MutableArray>>,
        validity: Option<MutableBitmap>,
    ) -> PolarsResult<Self> {
        check(&dtype, length, &values, validity.as_ref().map(|x| x.len()))?;
        Ok(Self {
            dtype,
            length,
            values,
            validity,
        })
    }

    /// Extract the low-end APIs from the [`MutableStructArray`].
    pub fn into_inner(
        self,
    ) -> (
        ArrowDataType,
        usize,
        Vec<Box<dyn MutableArray>>,
        Option<MutableBitmap>,
    ) {
        (self.dtype, self.length, self.values, self.validity)
    }

    /// The values
    pub fn values(&self) -> &Vec<Box<dyn MutableArray>> {
        &self.values
    }
}

impl MutableStructArray {
    /// Reserves `additional` entries.
    pub fn reserve(&mut self, additional: usize) {
        for v in &mut self.values {
            v.reserve(additional);
        }
        if let Some(x) = self.validity.as_mut() {
            x.reserve(additional)
        }
    }

    /// Call this once for each "row" of children you push.
    pub fn push(&mut self, valid: bool) {
        match &mut self.validity {
            Some(validity) => validity.push(valid),
            None => match valid {
                true => (),
                false => self.init_validity(),
            },
        };
        self.length += 1;
    }

    fn push_null(&mut self) {
        for v in &mut self.values {
            v.push_null();
        }
        self.push(false);
    }

    fn init_validity(&mut self) {
        let mut validity = MutableBitmap::with_capacity(self.values.capacity());
        let len = self.len();
        if len > 0 {
            validity.extend_constant(len, true);
            validity.set(len - 1, false);
        }
        self.validity = Some(validity)
    }

    /// Converts itself into an [`Array`].
    pub fn into_arc(self) -> Arc<dyn Array> {
        let a: StructArray = self.into();
        Arc::new(a)
    }

    /// Shrinks the capacity of the [`MutableStructArray`] to fit its current length.
    pub fn shrink_to_fit(&mut self) {
        for v in &mut self.values {
            v.shrink_to_fit();
        }
        if let Some(validity) = self.validity.as_mut() {
            validity.shrink_to_fit()
        }
    }
}

impl MutableArray for MutableStructArray {
    fn len(&self) -> usize {
        self.length
    }

    fn validity(&self) -> Option<&MutableBitmap> {
        self.validity.as_ref()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        StructArray::new(
            self.dtype.clone(),
            self.length,
            std::mem::take(&mut self.values)
                .into_iter()
                .map(|mut v| v.as_box())
                .collect(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .boxed()
    }

    fn as_arc(&mut self) -> Arc<dyn Array> {
        StructArray::new(
            self.dtype.clone(),
            self.length,
            std::mem::take(&mut self.values)
                .into_iter()
                .map(|mut v| v.as_box())
                .collect(),
            std::mem::take(&mut self.validity).map(|x| x.into()),
        )
        .arced()
    }

    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn as_any(&self) -> &dyn std::any::Any {
        self
    }

    fn as_mut_any(&mut self) -> &mut dyn std::any::Any {
        self
    }

    fn push_null(&mut self) {
        self.push_null()
    }

    fn shrink_to_fit(&mut self) {
        self.shrink_to_fit()
    }

    fn reserve(&mut self, additional: usize) {
        self.reserve(additional)
    }
}
