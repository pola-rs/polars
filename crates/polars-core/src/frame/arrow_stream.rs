use polars_arrow::array::StructArray;
use polars_arrow::ffi::{ArrowArrayStream, export_iterator};
use polars_arrow::record_batch::RecordBatch;

use crate::prelude::*;

impl DataFrame {
    /// Export this frame as an Arrow C stream with one struct array per record batch of
    /// [`DataFrame::try_iter_chunks`], converted as the stream is read.
    ///
    /// If the columns' chunks are not aligned, or a column holds Categorical or Enum values, the
    /// frame is rechunked first; otherwise its chunks are exported as they are.
    pub fn into_arrow_stream(mut self, compat_level: CompatLevel) -> ArrowArrayStream {
        self.align_chunks_for_arrow_export_par();
        let schema = self.schema().to_arrow(compat_level);
        batches_to_arrow_stream(self.try_iter_chunks(compat_level, false), schema)
    }

    /// Whether [`DataFrame::align_chunks_for_arrow_export`] would rechunk.
    pub fn should_rechunk_for_arrow_export(&self) -> bool {
        self.should_rechunk()
            || (self.max_n_chunks() > 1
                && self
                    .columns()
                    .iter()
                    .any(|c| c.dtype().contains_categoricals() || c.dtype().contains_enums()))
    }

    /// Align the chunks for an Arrow export, and rechunk if a column holds Categorical or Enum
    /// values, as each chunk would otherwise carry its own dictionary.
    pub fn align_chunks_for_arrow_export(&mut self) -> &mut Self {
        if self.should_rechunk_for_arrow_export() {
            self.rechunk_mut();
        }
        self
    }

    /// Like [`DataFrame::align_chunks_for_arrow_export`], but rechunks in parallel.
    pub fn align_chunks_for_arrow_export_par(&mut self) -> &mut Self {
        if self.should_rechunk_for_arrow_export() {
            self.rechunk_mut_par();
        }
        self
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
