#[cfg(feature = "python")]
pub mod python;

#[derive(Debug, Clone, PartialEq, Eq, Hash)]
#[cfg_attr(feature = "serde", derive(serde::Serialize, serde::Deserialize))]
#[cfg_attr(feature = "dsl-schema", derive(schemars::JsonSchema))]
pub enum ExternalReaderBuilder {
    #[cfg(feature = "python")]
    Python(python::PythonFileReaderBuilder),
    /// Unimplemented. Placeholder to avoid empty enum when Python feature is
    /// disabled.
    Rust(()),
}
