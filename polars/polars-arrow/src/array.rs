use arrow::array::ListArray;

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
