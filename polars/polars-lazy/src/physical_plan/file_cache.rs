use crate::prelude::*;
use polars_core::prelude::*;
use std::path::PathBuf;
use std::sync::{Arc, Mutex};

#[derive(Clone)]
#[allow(clippy::type_complexity)]
pub(crate) struct FileCache {
    // (path, predicate) -> (read_count, df)
    inner: Arc<Mutex<PlHashMap<(PathBuf, Option<Expr>), (FileCount, DataFrame)>>>,
}

impl FileCache {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Default::default())),
        }
    }

    pub(crate) fn read<F>(
        &self,
        key: (PathBuf, Option<Expr>),
        total_read_count: FileCount,
        reader: &mut F,
    ) -> Result<DataFrame>
    where
        F: FnMut() -> Result<DataFrame>,
    {
        if total_read_count == 1 {
            reader()
        } else {
            let mut mapping = self.inner.lock().unwrap();
            let (file_count, df_state) = mapping
                .entry(key)
                .or_insert_with(|| (0, Default::default()));

            // initialize df
            if *file_count == 0 {
                *df_state = reader()?;
            }
            *file_count += 1;

            // remove dataframe from memory
            if *file_count == total_read_count {
                Ok(std::mem::take(df_state))
            } else {
                Ok(df_state.clone())
            }
        }
    }
}
