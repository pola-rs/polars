use std::fmt::Debug;
use std::ops::{Deref, DerefMut};

use polars_async::ASYNC;
use polars_core::frame::DataFrame;
use polars_io::ipc::{IpcCompression, IpcReader, IpcWriter};
use polars_io::{SerReader, SerWriter};
use polars_utils::compression::ZstdLevel;

use crate::spill_context::ParameterFreeSpillContext;
use crate::spill_file::SpillFile;
use crate::{BYTES_SPILLED_TO_DISK, PinnedMut, PinnedRef, SpillToken, Spillable, memory_manager};

impl Spillable for DataFrame {
    // TODO: just a dummy spill for now. Boxed to reduce size.
    type Spilled = SpillFile;

    fn estimate_byte_size(&self) -> usize {
        self.estimated_size()
    }

    async fn spill(&self, context_id: &str) -> Self::Spilled {
        let mut df = self.clone();
        let context_id = context_id.to_owned();

        // Encode in the current task (on computational executor).
        let mut buf = Vec::new();

        let mut writer = IpcWriter::new(&mut buf).with_parallel(false);
        let clvl = polars_config::config().ooc_spill_compression_level();
        if clvl > 0 {
            let zstd_lvl = ZstdLevel::try_new(clvl.try_into().unwrap()).unwrap();
            writer = writer.with_compression(Some(IpcCompression::ZSTD(zstd_lvl)));
        }
        writer
            .finish(&mut df)
            .unwrap_or_else(|e| panic!("failed to encode spill file for '{context_id}': {e}",));

        // Do file creation / writing on tokio.
        ASYNC
            .spawn(async move {
                let size = buf.len() as u64;
                let spill_file = SpillFile::new(&context_id, "ipc", size);
                if BYTES_SPILLED_TO_DISK.fetch_add(size).saturating_add(size)
                    > polars_config::config().ooc_disk_budget_bytes()
                {
                    spill_file.creation_aborted();
                    polars_error::abort::polars_abort_ooc_out_of_disk();
                }
                tokio::fs::write(spill_file.path(), buf)
                    .await
                    .unwrap_or_else(|e| {
                        panic!(
                            "failed to create spill file '{}': {e}",
                            spill_file.path().display()
                        )
                    });
                spill_file
            })
            .await
            .unwrap()
    }

    async fn unspill(location: &Self::Spilled) -> Self {
        let path = location.path().to_owned();
        ASYNC
            .spawn_blocking(move || {
                let file = std::fs::File::open(&path).unwrap_or_else(|e| {
                    panic!("failed to open spill file {:?}: {e}", path.display())
                });
                IpcReader::new(file).finish().unwrap_or_else(|e| {
                    panic!("failed to read spill file {:?}: {e}", path.display())
                })
            })
            .await
            .unwrap()
    }
}

pub struct SpillFrame {
    token: SpillToken<DataFrame>,
    height: usize,
}

impl AsRef<SpillToken<DataFrame>> for SpillFrame {
    fn as_ref(&self) -> &SpillToken<DataFrame> {
        &self.token
    }
}

impl SpillFrame {
    pub fn new_unregistered(df: DataFrame) -> Self {
        let height = df.height();
        let token = SpillToken::new(df);
        Self { token, height }
    }

    pub async fn new<C: ParameterFreeSpillContext>(df: DataFrame, ctx: &C) -> Self {
        let slf = Self::new_unregistered(df);
        ctx.register(&slf);
        memory_manager().spill().await;
        slf
    }

    pub fn new_blocking<C: ParameterFreeSpillContext>(df: DataFrame, ctx: &C) -> Self {
        let slf = Self::new_unregistered(df);
        ctx.register(&slf);
        memory_manager().spill_blocking();
        slf
    }

    /// The height of the contained DataFrame. Does not need to unspill DataFrame.
    pub fn height(&self) -> usize {
        self.height
    }

    /// Get a reference to the underlying DataFrame, returning None if it was spilled.
    pub fn try_get(&self) -> Option<PinnedRef<'_, DataFrame>> {
        self.token.try_get()
    }

    /// Get a reference to the underlying DataFrame, unspilling it if it
    /// was spilled.
    pub async fn get(&self) -> PinnedRef<'_, DataFrame> {
        self.token.get().await
    }

    /// Blocking version of get.
    pub fn get_blocking(&self) -> PinnedRef<'_, DataFrame> {
        self.token.get_blocking()
    }

    /// Get a mutable reference to the underlying DataFrame, unspilling it if it
    /// was spilled.
    pub async fn get_mut(&mut self) -> PinnedFrameMut<'_> {
        PinnedFrameMut {
            inner: self.token.get_mut().await,
            height: &mut self.height,
        }
    }

    /// Blocking version of get_mut.
    pub fn get_mut_blocking(&mut self) -> PinnedFrameMut<'_> {
        PinnedFrameMut {
            inner: self.token.get_mut_blocking(),
            height: &mut self.height,
        }
    }

    /// Consumes this SpillFrame, unspilling it if it were spilled.
    pub async fn into_df(self) -> DataFrame {
        self.token.into_inner().await
    }

    /// Blocking version of into_df.
    pub fn into_df_blocking(self) -> DataFrame {
        self.token.into_inner_blocking()
    }
}

impl Debug for SpillFrame {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        let mut s = f.debug_struct("SpillFrame");
        match self.token.try_get() {
            Some(df) => s.field("df", &*df),
            None => s.field("df", &"spilled"),
        };
        s.finish()
    }
}

pub struct PinnedFrameMut<'a> {
    height: &'a mut usize,
    inner: PinnedMut<'a, DataFrame>,
}

impl<'a> Deref for PinnedFrameMut<'a> {
    type Target = DataFrame;

    fn deref(&self) -> &Self::Target {
        &self.inner
    }
}

impl<'a> DerefMut for PinnedFrameMut<'a> {
    fn deref_mut(&mut self) -> &mut Self::Target {
        &mut self.inner
    }
}

impl<'a> Drop for PinnedFrameMut<'a> {
    fn drop(&mut self) {
        *self.height = self.inner.height();
    }
}
