use super::primitive::ChunkedArray;
use arrow::datatypes::ArrowPrimitiveType;
use arrow::datatypes::Field;
use std::any::Any;
use std::fmt;

pub trait Series: fmt::Debug + Send + Sync {
    fn field(&self) -> &Field;
    fn as_any(&self) -> &dyn Any;
}

pub type SeriesRef = dyn Series;

impl<T> Series for ChunkedArray<T>
where
    T: ArrowPrimitiveType + Send + Sync,
{
    fn field(&self) -> &Field {
        &self.field
    }

    fn as_any(&self) -> &dyn Any {
        self
    }
}
