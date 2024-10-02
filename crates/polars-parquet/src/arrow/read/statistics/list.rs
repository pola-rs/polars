use arrow::array::*;
use arrow::datatypes::ArrowDataType;
use arrow::offset::Offsets;
use polars_error::PolarsResult;

use super::make_mutable;

#[derive(Debug)]
pub struct DynMutableListArray {
    dtype: ArrowDataType,
    pub inner: Box<dyn MutableArray>,
}

impl DynMutableListArray {
    pub fn try_with_capacity(dtype: ArrowDataType, capacity: usize) -> PolarsResult<Self> {
        let inner = match dtype.to_logical_type() {
            ArrowDataType::List(inner)
            | ArrowDataType::LargeList(inner)
            | ArrowDataType::FixedSizeList(inner, _) => inner.dtype(),
            _ => unreachable!(),
        };
        let inner = make_mutable(inner, capacity)?;

        Ok(Self { dtype, inner })
    }
}

impl MutableArray for DynMutableListArray {
    fn dtype(&self) -> &ArrowDataType {
        &self.dtype
    }

    fn len(&self) -> usize {
        self.inner.len()
    }

    fn validity(&self) -> Option<&arrow::bitmap::MutableBitmap> {
        self.inner.validity()
    }

    fn as_box(&mut self) -> Box<dyn Array> {
        let inner = self.inner.as_box();

        match self.dtype.to_logical_type() {
            ArrowDataType::List(_) => {
                let offsets =
                    Offsets::try_from_lengths(std::iter::repeat(1).take(inner.len())).unwrap();
                Box::new(ListArray::<i32>::new(
                    self.dtype.clone(),
                    offsets.into(),
                    inner,
                    None,
                ))
            },
            ArrowDataType::LargeList(_) => {
                let offsets =
                    Offsets::try_from_lengths(std::iter::repeat(1).take(inner.len())).unwrap();
                Box::new(ListArray::<i64>::new(
                    self.dtype.clone(),
                    offsets.into(),
                    inner,
                    None,
                ))
            },
            ArrowDataType::FixedSizeList(field, _) => Box::new(FixedSizeListArray::new(
                ArrowDataType::FixedSizeList(field.clone(), inner.len()),
                1,
                inner,
                None,
            )),
            _ => unreachable!(),
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
