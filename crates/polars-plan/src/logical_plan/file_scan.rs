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
        cloud_options: Option<CloudOptions>,
        #[cfg_attr(feature = "serde", serde(skip))]
        metadata: Option<Arc<FileMetaData>>,
    },
    #[cfg(feature = "ipc")]
    Ipc { options: IpcScanOptions },
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
            (FileScan::Ipc { options: l }, FileScan::Ipc { options: r }) => l == r,
            _ => false,
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

    pub(crate) fn skip_rows(&self) -> usize {
        #[allow(unreachable_patterns)]
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { options } => options.skip_rows,
            _ => 0,
        }
    }

    pub(crate) fn sort_projection(&self, _file_options: &FileScanOptions) -> bool {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => true,
            #[cfg(feature = "ipc")]
            Self::Ipc { .. } => _file_options.row_count.is_some(),
            #[cfg(feature = "parquet")]
            Self::Parquet { .. } => _file_options.row_count.is_some(),
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
