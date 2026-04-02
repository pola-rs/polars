#[derive(Clone, Copy, PartialEq, Eq, Debug, Default, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum SyncOnCloseType {
    /// Don't call sync on close.
    #[default]
    None,

    /// Sync only the file contents.
    Data,
    /// Synce the file contents and the metadata.
    All,
}
