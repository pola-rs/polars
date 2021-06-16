use arrow::{
    array::{ArrayRef, ListArray, Utf8Array},
    datatypes::DataType,
};

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

impl ValueSize for Utf8Array<i64> {
    fn get_values_size(&self) -> usize {
        self.values().len()
    }
}

impl ValueSize for ArrayRef {
    fn get_values_size(&self) -> usize {
        match self.data_type() {
            DataType::LargeUtf8 => self
                .as_any()
                .downcast_ref::<Utf8Array<i64>>()
                .unwrap()
                .get_values_size(),
            DataType::LargeList(_) => self
                .as_any()
                .downcast_ref::<ListArray<i64>>()
                .unwrap()
                .get_values_size(),
            _ => unimplemented!(),
        }
    }
}
