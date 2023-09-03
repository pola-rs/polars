use super::*;

impl SeriesOpsTime for Wrap<CategoricalChunked> {
    fn ops_time_dtype(&self) -> &DataType {
        self.0.dtype()
    }
}
