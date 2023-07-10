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
    Ipc { options: IpcScanOptionsInner },
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

    pub(crate) fn sort_projection(&self) -> bool {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { .. } => true,
            #[cfg(feature = "ipc")]
            Self::Ipc { options, .. } => options.row_count.is_some(),
            #[cfg(feature = "parquet")]
            Self::Parquet { options, .. } => options.row_count.is_some(),
        }
    }
}
