use std::path::PathBuf;

use polars_core::prelude::*;
use polars_lazy::prelude::*;
use rayon::prelude::*;

enum SourceState {
    Finished,
    HasMore(DataFrame),
}

trait Source {
    fn next() -> SourceState;
}

struct CsvSource {
    pub path: PathBuf,
    pub schema: SchemaRef,
    pub options: CsvParserOptions,
}

impl Source for CsvSource {}
