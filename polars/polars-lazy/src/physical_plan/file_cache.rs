use std::sync::Mutex;

use polars_core::prelude::*;
#[cfg(any(feature = "parquet", feature = "csv-file", feature = "ipc"))]
use polars_plan::logical_plan::FileFingerPrint;

use crate::prelude::*;

#[derive(Clone)]
pub(crate) struct FileCache {
    // (path, predicate) -> (read_count, df)
    inner: Arc<PlHashMap<FileFingerPrint, Mutex<(FileCount, DataFrame)>>>,
}

impl FileCache {
    pub(super) fn new(finger_prints: Option<Vec<FileFingerPrint>>) -> Self {
        let inner = match finger_prints {
            None => Arc::new(Default::default()),
            Some(fps) => {
                let mut mapping = PlHashMap::with_capacity(fps.len());
                for fp in fps {
                    mapping.insert(fp, Mutex::new((0, Default::default())));
                }
                Arc::new(mapping)
            }
        };

        Self { inner }
    }

    #[cfg(debug_assertions)]
    pub(crate) fn assert_empty(&self) {
        for (_, guard) in self.inner.iter() {
            let state = guard.lock().unwrap();
            assert!(state.1.is_empty());
        }
    }

    pub(crate) fn read<F>(
        &self,
        finger_print: FileFingerPrint,
        total_read_count: FileCount,
        reader: &mut F,
    ) -> PolarsResult<DataFrame>
    where
        F: FnMut() -> PolarsResult<DataFrame>,
    {
        if total_read_count == 1 {
            if total_read_count == 0 {
                eprintln!("we have hit an unexpected branch, please open an issue")
            }
            reader()
        } else {
            // should exist
            let guard = self.inner.get(&finger_print).unwrap();
            let mut state = guard.lock().unwrap();

            // initialize df
            if state.0 == 0 {
                state.1 = reader()?;
            }
            state.0 += 1;

            // remove dataframe from memory
            if state.0 == total_read_count {
                Ok(std::mem::take(&mut state.1))
            } else {
                Ok(state.1.clone())
            }
        }
    }
}
