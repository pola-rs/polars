use std::fs::File;
use std::path::Path;

use polars_core::prelude::*;
use polars_io::ipc::IpcReader;
use polars_io::SerReader;

use crate::operators::{DataChunk, PExecutionContext, Source, SourceResult};

/// Reads the whole file in one pass
pub struct IpcSourceOneShot {
    reader: Option<IpcReader<File>>,
}

impl IpcSourceOneShot {
    #[allow(unused_variables)]
    pub(crate) fn new(path: &Path) -> PolarsResult<Self> {
        let file = polars_utils::open_file(path)?;
        let reader = Some(IpcReader::new(file));

        Ok(IpcSourceOneShot { reader })
    }
}

impl Source for IpcSourceOneShot {
    fn get_batches(&mut self, _context: &PExecutionContext) -> PolarsResult<SourceResult> {
        if self.reader.is_none() {
            Ok(SourceResult::Finished)
        } else {
            let df = self.reader.take().unwrap().finish()?;
            Ok(SourceResult::GotMoreData(vec![DataChunk::new(0, df)]))
        }
    }
    fn fmt(&self) -> &str {
        "ipc-one-shot"
    }
}
