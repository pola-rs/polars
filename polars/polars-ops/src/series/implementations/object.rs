use super::*;

impl<T: PolarsObject> SeriesOps for Wrap<ObjectChunked<T>> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
