use arrow::array::*;
use arrow::datatypes::DataType as ArrowDataType;
use crate::sort_field::SortField;

use super::*;

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
