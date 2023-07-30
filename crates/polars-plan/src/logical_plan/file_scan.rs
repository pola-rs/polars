use super::*;

#[derive(Clone, Debug, IntoStaticStr, PartialEq)]
#[cfg_attr(feature = "serde", derive(Serialize, Deserialize))]
pub enum FileScan {
    #[cfg(feature = "csv")]
    Csv { options: CsvParserOptions },
    #[cfg(feature = "parquet")]
    Parquet {
        options: ParquetOptions,
        cloud_options: Option<CloudOptions>,
    },
    #[cfg(feature = "ipc")]
    Ipc { options: IpcScanOptions },
}

impl FileScan {
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
