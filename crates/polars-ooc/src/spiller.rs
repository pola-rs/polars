use polars_config::SpillFormat;
use polars_core::prelude::DataFrame;

use crate::token::Token;

pub struct Spiller {
    #[allow(dead_code)]
    format: SpillFormat,
}

impl Spiller {
    pub fn new(format: SpillFormat) -> Self {
        Self { format }
    }

    /// Spill a DataFrame to disk.
    #[allow(dead_code)]
    pub fn spill(&self, _token: &Token, _df: DataFrame) {
        unimplemented!("spilling to disk")
    }

    /// Load a previously spilled DataFrame from disk.
    pub fn load(&self, _token: &Token) -> DataFrame {
        unimplemented!("loading spilled data from disk")
    }

    /// Load a previously spilled DataFrame from disk (blocking).
    pub fn load_blocking(&self, _token: &Token) -> DataFrame {
        unimplemented!("loading spilled data from disk")
    }

    /// Best-effort delete of a spill file.
    pub fn delete(&self, _token: &Token) {
        unimplemented!("deleting spill file")
    }
}
