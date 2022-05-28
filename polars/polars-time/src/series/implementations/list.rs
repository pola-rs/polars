use super::*;

impl SeriesOpsTime for Wrap<ListChunked> {
    fn ops_time_dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
