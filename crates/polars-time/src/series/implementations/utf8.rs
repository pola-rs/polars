use super::*;

impl SeriesOpsTime for Wrap<Utf8Chunked> {
    fn ops_time_dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
