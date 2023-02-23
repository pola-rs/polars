use arrow::array::*;
use arrow::datatypes::*;

use super::*;

/// Options that define the sort order of a given column
pub struct SortOptions {
    /// Whether to sort in descending order
    pub descending: bool,
    /// Whether to sort nulls first
    pub nulls_first: bool,
}

pub struct SortField {
    /// Sort options
    options: SortOptions,
    /// Data type
    data_type: DataType,
}

pub struct RowConverter {
    fields: Vec<SortField>,
}

impl RowConverter {
    pub fn new(fields: Vec<SortField>) -> Self {
        Self { fields }
    }

    pub fn convert_columns(&self, columns: &[ArrayRef]) {
        assert_eq!(self.fields.len(), columns.len());
    }
}
