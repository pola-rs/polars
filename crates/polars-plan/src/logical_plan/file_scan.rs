use std::hash::{Hash, Hasher};

#[cfg(feature = "parquet")]
use polars_parquet::write::FileMetaData;

use super::*;

#[derive(Clone, Debug, IntoStaticStr)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FileScan {
    #[cfg(feature = "csv")]
    Csv { options: CsvParserOptions },
    #[cfg(feature = "parquet")]
    Parquet {
        options: ParquetOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        #[cfg_attr(feature = "serde", serde(skip))]
        metadata: Option<Arc<FileMetaData>>,
    },
    #[cfg(feature = "ipc")]
    Ipc {
        options: IpcScanOptions,
        cloud_options: Option<polars_io::cloud::CloudOptions>,
        #[cfg_attr(feature = "serde", serde(skip))]
        metadata: Option<arrow::io::ipc::read::FileMetadata>,
    },
    #[cfg_attr(feature = "serde", serde(skip))]
    Anonymous {
        options: Arc<AnonymousScanOptions>,
        function: Arc<dyn AnonymousScan>,
    },
}

impl PartialEq for FileScan {
    fn eq(&self, other: &Self) -> bool {
        match (self, other) {
            #[cfg(feature = "csv")]
            (FileScan::Csv { options: l }, FileScan::Csv { options: r }) => l == r,
            #[cfg(feature = "parquet")]
            (
                FileScan::Parquet {
                    options: opt_l,
                    cloud_options: c_l,
                    ..
                },
                FileScan::Parquet {
                    options: opt_r,
                    cloud_options: c_r,
                    ..
                },
            ) => opt_l == opt_r && c_l == c_r,
            #[cfg(feature = "ipc")]
            (
                FileScan::Ipc {
                    options: l,
                    cloud_options: c_l,
                    ..
                },
                FileScan::Ipc {
                    options: r,
                    cloud_options: c_r,
                    ..
                },
            ) => l == r && c_l == c_r,
            _ => false,
        }
    }
}

impl Eq for FileScan {}

impl Hash for FileScan {
    fn hash<H: Hasher>(&self, state: &mut H) {
        std::mem::discriminant(self).hash(state);
        match self {
            #[cfg(feature = "csv")]
            FileScan::Csv { options } => options.hash(state),
            #[cfg(feature = "parquet")]
            FileScan::Parquet {
                options,
                cloud_options,
                metadata: _,
            } => {
                options.hash(state);
                cloud_options.hash(state)
            },
            #[cfg(feature = "ipc")]
            FileScan::Ipc {
                options,
                cloud_options,
                metadata: _,
            } => {
                options.hash(state);
                cloud_options.hash(state);
            },
            FileScan::Anonymous { options, .. } => options.hash(state),
        }
    }
}

impl FileScan {
    pub(crate) fn remove_metadata(&mut self) {
        match self {
            #[cfg(feature = "parquet")]
            Self::Parquet { metadata, .. } => {
                *metadata = None;
            },
            _ => {},
        }
    }

    pub(crate) fn sort_projection(&self, _file_options: &FileScanOptions) -> bool {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => true,
            #[cfg(feature = "ipc")]
            Self::Ipc { .. } => _file_options.row_index.is_some(),
            #[cfg(feature = "parquet")]
            Self::Parquet { .. } => _file_options.row_index.is_some(),
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }

    pub fn streamable(&self) -> bool {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => true,
            #[cfg(feature = "ipc")]
            Self::Ipc { .. } => false,
            #[cfg(feature = "parquet")]
            Self::Parquet { .. } => true,
            #[allow(unreachable_patterns)]
            _ => false,
        }
    }
}
