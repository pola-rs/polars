use crate::array::{Array, ArrayRef, BinaryArray, FixedSizeListArray, ListArray, Utf8Array};
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;

pub trait ValueSize {
    /// Useful for a Utf8 or a List to get underlying value size.
    /// During a rechunk this is handy
    fn get_values_size(&self) -> usize;
}

impl ValueSize for ListArray<i64> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for FixedSizeListArray {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for Utf8Array<i64> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl<O: Offset> ValueSize for BinaryArray<O> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for ArrayRef {
    fn get_values_size(&self) -> usize {
        match self.data_type() {
            ArrowDataType::LargeUtf8 => self
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .get_values_size(),
            ArrowDataType::FixedSizeList(_, _) => self
                .as_any()
                .downcast_ref::<FixedSizeListArray>()
                .unwrap()
                .get_values_size(),
            ArrowDataType::LargeList(_) => self
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap()
                .get_values_size(),
            ArrowDataType::LargeBinary => self
                .as_any()
                .downcast_ref::<BinaryArray<i64>>()
                .unwrap()
                .get_values_size(),
            _ => unimplemented!(),
        }
    }
}
