//! A file reader builder of an I/O plugin (`polars.io_plugin.file_reader.v1`).
use std::fmt;
use std::hash::{Hash, Hasher};

use polars_error::{PolarsResult, polars_err};
use polars_io_ext_ffi::common::{FfiBuf, FfiSlice};
use polars_io_ext_ffi::file_reader_v1::{BuilderHandle, Capabilities, Plugin};
use polars_utils::aliases::PlIndexMap;
use polars_utils::pl_str::PlSmallStr;

use crate::io_plugin_ffi::{ffi_err_to_polars, file_reader_v1};

/// One reference to a builder handle of the plugin `package`.
pub struct FfiFileReaderBuilder {
    package: PlSmallStr,
    plugin: &'static Plugin,
    handle: BuilderHandle,
}

// The contract makes builder handles thread-safe.
unsafe impl Send for FfiFileReaderBuilder {}
unsafe impl Sync for FfiFileReaderBuilder {}

impl FfiFileReaderBuilder {
    /// Takes ownership of one reference to `handle`, a builder of `package`.
    ///
    /// # Safety
    /// `handle` must be a live builder handle of `package`'s `file_reader.v1` contract.
    pub unsafe fn from_owned_handle(
        package: PlSmallStr,
        handle: BuilderHandle,
    ) -> PolarsResult<Self> {
        let plugin = file_reader_v1(&package)?;
        Ok(Self {
            package,
            plugin,
            handle,
        })
    }

    pub fn package(&self) -> &PlSmallStr {
        &self.package
    }

    pub fn plugin(&self) -> &'static Plugin {
        self.plugin
    }

    pub fn handle(&self) -> BuilderHandle {
        self.handle
    }

    pub fn name(&self) -> PlSmallStr {
        let name: FfiBuf = unsafe { (self.plugin.builder_name)(self.handle) };
        name.to_string_lossy().into()
    }

    pub fn fingerprint(&self) -> u64 {
        unsafe { (self.plugin.builder_fingerprint)(self.handle) }
    }

    pub fn capabilities(&self) -> PolarsResult<Capabilities> {
        unsafe { (self.plugin.builder_capabilities)(self.handle) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &self.package))
    }

    pub fn explain_properties(&self) -> PolarsResult<PlIndexMap<String, String>> {
        let props = unsafe { (self.plugin.builder_explain)(self.handle) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &self.package))?;
        Ok(props
            .as_slice()
            .chunks_exact(2)
            .map(|kv| (kv[0].to_string_lossy(), kv[1].to_string_lossy()))
            .collect())
    }

    fn serialize_bytes(&self) -> PolarsResult<FfiBuf> {
        unsafe { (self.plugin.builder_serialize)(self.handle) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &self.package))
    }

    fn deserialize_bytes(package: PlSmallStr, bytes: &[u8]) -> PolarsResult<Self> {
        let plugin = file_reader_v1(&package)?;
        let handle = unsafe { (plugin.builder_deserialize)(FfiSlice::new(bytes)) }
            .into_result()
            .map_err(|e| ffi_err_to_polars(e, &package))?;
        Ok(Self {
            package,
            plugin,
            handle,
        })
    }
}

impl Clone for FfiFileReaderBuilder {
    fn clone(&self) -> Self {
        Self {
            package: self.package.clone(),
            plugin: self.plugin,
            handle: unsafe { (self.plugin.builder_clone)(self.handle) },
        }
    }
}

impl Drop for FfiFileReaderBuilder {
    fn drop(&mut self) {
        unsafe { (self.plugin.builder_release)(self.handle) }
    }
}

impl fmt::Debug for FfiFileReaderBuilder {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.debug_struct("FfiFileReaderBuilder")
            .field("package", &self.package)
            .field("name", &self.name())
            .finish()
    }
}

/// Builders of one plugin with the same fingerprint read the same data.
impl PartialEq for FfiFileReaderBuilder {
    fn eq(&self, other: &Self) -> bool {
        self.package == other.package && self.fingerprint() == other.fingerprint()
    }
}

impl Eq for FfiFileReaderBuilder {}

impl Hash for FfiFileReaderBuilder {
    fn hash<H: Hasher>(&self, state: &mut H) {
        self.package.hash(state);
        self.fingerprint().hash(state);
    }
}

#[cfg(feature = "serde")]
mod _serde {
    use serde::{Deserialize, Deserializer, Serialize, Serializer};

    use super::*;

    #[derive(Serialize, Deserialize)]
    struct Serialized {
        package: PlSmallStr,
        #[serde(with = "serde_bytes_vec")]
        bytes: Vec<u8>,
    }

    mod serde_bytes_vec {
        use serde::{Deserializer, Serializer};

        pub fn serialize<S: Serializer>(v: &[u8], s: S) -> Result<S::Ok, S::Error> {
            s.serialize_bytes(v)
        }

        pub fn deserialize<'de, D: Deserializer<'de>>(d: D) -> Result<Vec<u8>, D::Error> {
            polars_utils::pl_serialize::deserialize_map_bytes(d, |b| b.into_owned())
        }
    }

    impl Serialize for FfiFileReaderBuilder {
        fn serialize<S: Serializer>(&self, serializer: S) -> Result<S::Ok, S::Error> {
            use serde::ser::Error;
            let bytes = self.serialize_bytes().map_err(S::Error::custom)?;
            Serialized {
                package: self.package.clone(),
                bytes: bytes.to_vec(),
            }
            .serialize(serializer)
        }
    }

    impl<'de> Deserialize<'de> for FfiFileReaderBuilder {
        fn deserialize<D: Deserializer<'de>>(deserializer: D) -> Result<Self, D::Error> {
            use serde::de::Error;
            let s = Serialized::deserialize(deserializer)?;
            Self::deserialize_bytes(s.package, &s.bytes)
                .map_err(|e| D::Error::custom(polars_err!(ComputeError: "{e}")))
        }
    }
}
