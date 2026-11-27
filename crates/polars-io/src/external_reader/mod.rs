use polars_error::PolarsResult;
use polars_utils::aliases::PlIndexMap;

#[cfg(feature = "io_plugin_ffi")]
pub mod ffi;
#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ExternalReaderBuilder {
    #[cfg(feature = "python")]
    Python(python::PythonFileReaderBuilder),
    /// A reader of an I/O plugin, called through `polars.io_plugin.file_reader.v1`.
    #[cfg(feature = "io_plugin_ffi")]
    #[cfg_attr(feature = "dsl-schema", schemars(skip))]
    Ffi(ffi::FfiFileReaderBuilder),
    /// Unimplemented. Placeholder to avoid empty enum when Python feature is
    /// disabled.
    Rust(()),
}

impl ExternalReaderBuilder {
    pub fn explain_properties(&self) -> PolarsResult<PlIndexMap<String, String>> {
        match self {
            #[cfg(feature = "python")]
            ExternalReaderBuilder::Python(x) => x.explain_properties(),
            #[cfg(feature = "io_plugin_ffi")]
            ExternalReaderBuilder::Ffi(x) => x.explain_properties(),
            ExternalReaderBuilder::Rust(()) => unimplemented!(),
        }
    }

    /// Whether a negative slice may be resolved from the row counts of the sources. An I/O
    /// plugin counts with a statement of its own, so the rows may change before they are read.
    pub fn counts_match_reads(&self) -> bool {
        match self {
            #[cfg(feature = "io_plugin_ffi")]
            ExternalReaderBuilder::Ffi(_) => false,
            _ => true,
        }
    }
}
