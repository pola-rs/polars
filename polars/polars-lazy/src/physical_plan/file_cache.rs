use crate::prelude::file_caching::FileFingerPrint;
use crate::prelude::*;
use parking_lot::Mutex;
use polars_core::prelude::*;

#[derive(Clone)]
pub(crate) struct FileCache {
    // (path, predicate) -> (read_count, df)
    inner: Arc<Mutex<PlHashMap<FileFingerPrint, (FileCount, DataFrame)>>>,
}

impl FileCache {
    pub(super) fn new() -> Self {
        Self {
            inner: Arc::new(Mutex::new(Default::default())),
        }
    }
    pub(crate) fn read<F>(
        &self,
        finger_print: FileFingerPrint,
        total_read_count: FileCount,
        reader: &mut F,
    ) -> Result<DataFrame>
    where
        F: FnMut() -> Result<DataFrame>,
    {
        if total_read_count == 1 {
            if total_read_count == 0 {
                eprintln!("we have hit an unexpected branch, please open an issue")
            }
            reader()
        } else {
            let mut mapping = self.inner.lock();
            let (file_count, df_state) = mapping
                .entry(finger_print)
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
