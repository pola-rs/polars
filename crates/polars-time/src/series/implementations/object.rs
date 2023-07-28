use super::*;

impl<T: PolarsObject> SeriesOpsTime for Wrap<ObjectChunked<T>> {
    fn ops_time_dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
