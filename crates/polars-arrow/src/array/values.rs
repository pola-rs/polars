use crate::array::{
    ArrayRef, BinaryArray, BinaryViewArray, FixedSizeListArray, ListArray, Utf8Array, Utf8ViewArray,
};
use crate::datatypes::ArrowDataType;
use crate::offset::Offset;

pub trait ValueSize {
    /// Get the values size that is still "visible" to the underlying array.
    /// E.g. take the offsets into account.
    fn get_values_size(&self) -> usize;
}

impl ValueSize for ListArray<i64> {
    fn get_values_size(&self) -> usize {
        unsafe {
            // SAFETY:
            // invariant of the struct that offsets always has at least 2 members.
            let start = *self.offsets().get_unchecked(0) as usize;
            let end = *self.offsets().last() as usize;
            end - start
        }
    }
}

impl ValueSize for FixedSizeListArray {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for Utf8Array<i64> {
    fn get_values_size(&self) -> usize {
        unsafe {
            // SAFETY:
            // invariant of the struct that offsets always has at least 2 members.
            let start = *self.offsets().get_unchecked(0) as usize;
            let end = *self.offsets().last() as usize;
            end - start
        }
    }
}

impl<O: Offset> ValueSize for BinaryArray<O> {
    fn get_values_size(&self) -> usize {
        unsafe {
            // SAFETY:
            // invariant of the struct that offsets always has at least 2 members.
            let start = self.offsets().get_unchecked(0).to_usize();
            let end = self.offsets().last().to_usize();
            end - start
        }
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
            ArrowDataType::Utf8View => self
                .as_any()
                .downcast_ref::<Utf8ViewArray>()
                .unwrap()
                .total_bytes_len(),
            ArrowDataType::BinaryView => self
                .as_any()
                .downcast_ref::<BinaryViewArray>()
                .unwrap()
                .total_bytes_len(),
            _ => unimplemented!(),
        }
    }
}
