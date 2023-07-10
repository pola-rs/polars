use polars_io::RowCount;

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
    },
    #[cfg(feature = "ipc")]
    Ipc { options: IpcScanOptionsInner },
}

impl FileScan {
    pub(crate) fn with_columns(&self) -> Option<&Arc<Vec<String>>> {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { options } => options.with_columns.as_ref(),
            #[cfg(feature = "ipc")]
            Self::Ipc { options, .. } => options.with_columns.as_ref(),
            #[cfg(feature = "parquet")]
            Self::Parquet { options, .. } => options.with_columns.as_ref(),
        }
    }

    pub(crate) fn with_columns_mut(&mut self) -> &mut Option<Arc<Vec<String>>> {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { options } => &mut options.with_columns,
            #[cfg(feature = "ipc")]
            Self::Ipc { options, .. } => &mut options.with_columns,
            #[cfg(feature = "parquet")]
            Self::Parquet { options, .. } => &mut options.with_columns,
        }
    }

    pub(crate) fn row_count(&self) -> Option<&RowCount> {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { options } => options.row_count.as_ref(),
            #[cfg(feature = "ipc")]
            Self::Ipc { options, .. } => options.row_count.as_ref(),
            #[cfg(feature = "parquet")]
            Self::Parquet { options, .. } => options.row_count.as_ref(),
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

    pub(crate) fn n_rows(&self) -> Option<usize> {
        match self {
            #[cfg(feature = "csv")]
            Self::Csv { options } => options.n_rows,
            #[cfg(feature = "parquet")]
            Self::Parquet { options, .. } => options.n_rows,
            #[cfg(feature = "ipc")]
            Self::Ipc { options, .. } => options.n_rows,
        }
    }
}
