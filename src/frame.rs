use crate::series::series::SeriesRef;
use arrow::datatypes::Field;

struct DataFrame {
    columns: Vec<Box<SeriesRef>>,
}

impl DataFrame {
    fn fields(&self) -> Vec<&Field> {
        self.columns.iter().map(|s| s.field()).collect()
    }
}
