use polars_arrow::array::StructArray;
use polars_arrow::ffi::{ArrowArrayStream, export_iterator};
use polars_arrow::record_batch::RecordBatch;

use crate::prelude::*;

impl DataFrame {
    /// Export this frame as an Arrow C stream with one struct array per record batch of
    /// [`DataFrame::try_iter_chunks`], converted as the stream is read.
    ///
    /// Misaligned chunks are rechunked first; aligned ones are exported as they are.
    pub fn into_arrow_stream(mut self, compat_level: CompatLevel) -> ArrowArrayStream {
        self.align_chunks_par();
        let schema = self.schema().to_arrow(compat_level);
        batches_to_arrow_stream(self.try_iter_chunks(compat_level, false), schema)
    }
}

/// Export `batches`, whose columns have the fields of `schema`, as an Arrow C stream of struct
/// arrays, the way a C stream carries a table.
pub fn batches_to_arrow_stream(
    batches: impl Iterator<Item = PolarsResult<RecordBatch>> + Send + 'static,
    schema: ArrowSchema,
) -> ArrowArrayStream {
    let dtype = ArrowDataType::Struct(schema.into_iter_values().collect());
    let field = ArrowField::new(PlSmallStr::EMPTY, dtype.clone(), false);
    let arrays = batches.map(move |batch| {
        let batch = batch?;
        let array = StructArray::new(dtype.clone(), batch.len(), batch.into_arrays(), None);
        Ok(array.boxed())
    });
    export_iterator(Box::new(arrays), field)
}
