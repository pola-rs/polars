use polars_utils::IdxSize;

pub static POLARS_IPC_METADATA_KEY: &str = "__POLARS_IPC_METADATA";

#[derive(Debug, Clone, Default, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
pub struct PlIpcMetadata {
    /// Cumulative length including the current record batch.
    pub record_batch_cum_len: Vec<IdxSize>,
}

#[cfg(feature = "serde")]
impl PlIpcMetadata {
    /// Reads the Polars metadata out of an already parsed IPC footer.
    ///
    /// Returns `None` for a file that was not written by Polars.
    pub fn from_ipc_footer(metadata: &arrow::io::ipc::read::FileMetadata) -> Option<Self> {
        let raw = metadata.custom_metadata.as_ref()?.get(POLARS_IPC_METADATA_KEY)?;
        serde_json::from_str(raw).ok()
    }

    /// Total number of rows over all record batches.
    pub fn num_rows(&self) -> Option<IdxSize> {
        self.record_batch_cum_len.last().copied()
    }
}
