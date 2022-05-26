use super::*;

impl<T: PolarsObject> SeriesOpsTime for Wrap<ObjectChunked<T>> {
    fn dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
