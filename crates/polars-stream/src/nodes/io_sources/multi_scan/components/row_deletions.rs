use std::sync::Arc;

use arrow::bitmap::bitmask::BitMask;
use arrow::bitmap::{Bitmap, MutableBitmap};
use polars_core::frame::DataFrame;
use polars_core::prelude::{BooleanChunked, ChunkAgg, DataType, PlIndexMap};
use polars_core::schema::{Schema, SchemaRef};
use polars_core::utils::accumulate_dataframes_vertical_unchecked;
use polars_error::{PolarsResult, feature_gated};
use polars_io::cloud::CloudOptions;
use polars_plan::dsl::deletion::DeletionFilesList;
use polars_plan::dsl::{CastColumnsPolicy, ScanSource};
use polars_utils::format_pl_smallstr;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::plpath::PlPath;
use polars_utils::slice_enum::Slice;

use crate::async_executor::{self, AbortOnDropHandle, TaskPriority};
use crate::nodes::io_sources::multi_scan::reader_interface::builder::FileReaderBuilder;
use crate::nodes::io_sources::multi_scan::reader_interface::{BeginReadArgs, FileReaderCallbacks};
#[cfg(feature = "parquet")]
use crate::nodes::io_sources::parquet::builder::ParquetReaderBuilder;

#[derive(Clone)]
pub enum DeletionFilesProvider {
    None,

    #[cfg(feature = "parquet")]
    IcebergPositionDelete {
        paths: Arc<PlIndexMap<usize, Arc<[String]>>>,
        // Amortized allocations
        reader_builder: ParquetReaderBuilder,
        projected_schema: SchemaRef,
    },
}

impl DeletionFilesProvider {
    pub fn new(deletion_files: Option<DeletionFilesList>) -> Self {
        if deletion_files.is_none() {
            return Self::None;
        }

        match deletion_files.unwrap() {
            DeletionFilesList::IcebergPositionDelete(paths) => feature_gated!(
                "parquet",
                Self::IcebergPositionDelete {
                    paths,
                    reader_builder: ParquetReaderBuilder {
                        first_metadata: None,
                        options: Arc::new(polars_io::prelude::ParquetOptions {
                            schema: Some(Arc::new(Schema::from_iter([
                                (PlSmallStr::from_static("file_path"), DataType::String),
                                (PlSmallStr::from_static("pos"), DataType::Int64),
                            ]))),

                            parallel: polars_io::prelude::ParallelStrategy::Auto,
                            low_memory: false,
                            use_statistics: false,
                        }),
                    },
                    projected_schema: Arc::new(Schema::from_iter([
                        (PlSmallStr::from_static("file_path"), DataType::String),
                        (PlSmallStr::from_static("pos"), DataType::Int64),
                    ])),
                }
            ),
        }
    }

    pub fn spawn_row_deletions_init(
        &self,
        scan_source_idx: usize,
        cloud_options: Option<Arc<CloudOptions>>,
        num_pipelines: usize,
        verbose: bool,
    ) -> Option<RowDeletionsInit> {
        match self {
            Self::None => None,

            #[cfg(feature = "parquet")]
            Self::IcebergPositionDelete {
                paths,
                reader_builder,
                projected_schema,
            } => {
                let paths = paths.get(&scan_source_idx)?;

                if verbose {
                    eprintln!(
                        "[DeletionFilesProvider[Iceberg]]: scan_source_idx: {}, {} files",
                        scan_source_idx,
                        paths.len()
                    )
                }

                // We create the readers and immediately spawn off tasks to initialize all of them.
                let file_readers = paths
                    .iter()
                    .enumerate()
                    .map(|(deletion_file_idx, path)| {
                        let source = ScanSource::Path(PlPath::new(path));
                        let mut reader = reader_builder.build_file_reader(
                            source,
                            cloud_options.clone(),
                            deletion_file_idx,
                        );

                        if verbose {
                            eprintln!(
                                "[DeletionFilesProvider[Iceberg]]: scan_source_idx: {scan_source_idx}, \
                                deletion_file_idx: {deletion_file_idx}, \
                                deletion_file_path: {path}"
                            )
                        }

                        AbortOnDropHandle::new(async_executor::spawn(
                            TaskPriority::Low,
                            async move {
                                reader.initialize().await?;
                                PolarsResult::Ok(reader)
                            },
                        ))
                    })
                    .collect::<Vec<_>>();

                let projected_schema = projected_schema.clone();

                // We choose to load deletion files immediately during the initialization phase -
                // the main driver loop of the multi file may need to serially `.await` on this
                // between initializing readers when there is a slice.
                //
                // This does mean deletion file loads are tied to `NUM_READERS_PRE_INIT`, but this
                // should be fine as the size of the data should not be too big.
                let handle = AbortOnDropHandle::new(async_executor::spawn(
                    TaskPriority::Low,
                    async move {
                        let handles = file_readers
                            .into_iter()
                            .map(|init_fut| {
                                use crate::nodes::io_sources::multi_scan::components::projection::Projection;

                                let begin_read_args = BeginReadArgs {
                                    projection: Projection::Plain(projected_schema.clone()),
                                    row_index: None,
                                    pre_slice: None,
                                    predicate: None,
                                    cast_columns_policy: CastColumnsPolicy::ERROR_ON_MISMATCH,
                                    num_pipelines,
                                    callbacks: FileReaderCallbacks {
                                        file_schema_tx: None,
                                        n_rows_in_file_tx: None,
                                        row_position_on_end_tx: None,
                                    },
                                };

                                AbortOnDropHandle::new(async_executor::spawn(
                                    TaskPriority::Low,
                                    async move {
                                        let mut reader = init_fut.await?;

                                        let (mut rx, handle) =
                                            reader.begin_read(begin_read_args)?;

                                        let mut dfs = vec![];

                                        while let Ok(morsel) = rx.recv().await {
                                            dfs.push(morsel.into_df());
                                        }

                                        handle.await?;

                                        let df = accumulate_dataframes_vertical_unchecked(dfs);

                                        // Some quick testing on AWS Athena showed that it doesn't
                                        // write deletion files that reference multiple distinct
                                        // file paths, so we don't handle that for now.
                                        assert!(
                                            df.column("file_path")?.n_unique()? <= 1,
                                            "assertion failed: iceberg position delete file: \
                                            n_unique(data_file_paths) <= 1. \
                                            This is a bug, please open an issue"
                                        );

                                        let positions_col = df.column("pos")?.clone();
                                        let max_idx = usize::try_from(
                                            positions_col
                                                .as_materialized_series_maintain_scalar()
                                                .i64()
                                                .unwrap()
                                                .max()
                                                .unwrap_or(0),
                                        )
                                        .unwrap();

                                        PolarsResult::Ok((positions_col, max_idx))
                                    },
                                ))
                            })
                            .collect::<Vec<_>>();

                        let mut position_columns = Vec::with_capacity(handles.len());
                        let mut filter_mask_len: usize = 0;

                        for handle in handles {
                            let (positions_col, max_idx) = handle.await?;
                            filter_mask_len = filter_mask_len.max(max_idx.saturating_add(1));
                            position_columns.push(positions_col);
                        }

                        let mut filter_mask = MutableBitmap::from_len_set(filter_mask_len);

                        for c in position_columns {
                            for idx in c.as_materialized_series_maintain_scalar().i64().unwrap() {
                                let idx = usize::try_from(idx.unwrap()).unwrap();
                                filter_mask.set(idx, false);
                            }
                        }

                        let bitmap = filter_mask.freeze();

                        // Also trigger the bitcount to reduce blocking later down.
                        bitmap.unset_bits();
                        debug_assert!(bitmap.lazy_unset_bits().is_some());

                        let mask = BooleanChunked::from_bitmap(PlSmallStr::EMPTY, bitmap);
                        let mask = ExternalFilterMask::IcebergPositionDelete { mask };

                        if verbose {
                            let num_deleted_rows = mask.num_deleted_rows();
                            let max_index = mask.len().checked_sub(1);

                            eprintln!(
                                "[DeletionFilesProvider[Iceberg]]: \
                                scan_source_idx: {scan_source_idx}, \
                                num_deleted_rows: {num_deleted_rows}, \
                                max_index: {max_index:?}",
                            )
                        }

                        Ok(mask)
                    },
                ));

                Some(RowDeletionsInit::Initializing(handle))
            },
        }
    }
}

pub enum RowDeletionsInit {
    Initializing(AbortOnDropHandle<PolarsResult<ExternalFilterMask>>),

    /// If negative slice resolution happens we will already have the initialized filter mask
    /// much earlier in the pipeline, but the channel interface still requires a `RowDeletionsInit`,
    /// so this enum variant is used to hold the initialized mask.
    Initialized(ExternalFilterMask),
}

impl RowDeletionsInit {
    /// Loads the deletion information into a filter mask.
    pub async fn into_external_filter_mask(self) -> PolarsResult<ExternalFilterMask> {
        match self {
            Self::Initializing(handle) => handle.await,
            Self::Initialized(v) => Ok(v),
        }
    }
}

#[derive(Debug, Clone)]
pub enum ExternalFilterMask {
    /// Note: Iceberg positional deletes can have a mask length shorter than the actual data.
    IcebergPositionDelete { mask: BooleanChunked },
}

impl ExternalFilterMask {
    pub fn variant_name(&self) -> &'static str {
        use ExternalFilterMask::*;
        match self {
            IcebergPositionDelete { .. } => "IcebergPositionDelete",
        }
    }

    /// Human-friendly verbose log display.
    pub fn log_display(this: Option<&Self>) -> PlSmallStr {
        match this {
            None => PlSmallStr::from_static("None"),
            Some(mask) => {
                let mask_variant = mask.variant_name();
                let n = mask.num_deleted_rows();
                let s = if n == 1 { "" } else { "s" };
                format_pl_smallstr!("{mask_variant}(<{n} deletion{s}>)")
            },
        }
    }

    pub fn filter_df(&self, df: &mut DataFrame) -> PolarsResult<()> {
        match self {
            Self::IcebergPositionDelete { mask } => {
                if !mask.is_empty() {
                    *df = if mask.len() < df.height() {
                        accumulate_dataframes_vertical_unchecked([
                            df.slice(0, mask.len())._filter_seq(mask)?,
                            df.slice(i64::try_from(mask.len()).unwrap(), df.height() - mask.len()),
                        ])
                    } else {
                        df._filter_seq(mask)?
                    }
                }
            },
        }

        Ok(())
    }

    pub fn slice(&self, offset: usize, len: usize) -> Self {
        match self {
            Self::IcebergPositionDelete { mask } => {
                // This is not a valid offset, it's also a sentinel value from `RowCounter::MAX`.
                assert_ne!(offset, usize::MAX);
                let offset = offset.min(mask.len());
                let len = len.min(mask.len() - offset);

                let mask = mask.slice(i64::try_from(offset).unwrap(), len);

                Self::IcebergPositionDelete { mask }
            },
        }
    }

    pub fn num_deleted_rows(&self) -> usize {
        match self {
            Self::IcebergPositionDelete { mask } => mask
                .rechunk()
                .downcast_get(0)
                .unwrap()
                .values()
                .unset_bits(),
        }
    }

    /// Calculates the physical pre_slice that can be applied before performing row deletions.
    ///
    /// By default, a `pre_slice` is applied after rows are deleted. This function takes a `pre_slice`
    /// and translates it to `physical` positions (i.e. one that can be applied before row deletions).
    /// This is done by expanding the range of the slice to account for the deleted rows.
    ///
    /// This involves 2 `nth_set_bit` searches for offset and length.
    ///
    /// # Panics
    /// Panics if `slice` is negative.
    pub fn calc_physical_slice(&self, slice: Slice) -> Slice {
        let mask = self.get_mask();

        let phys_slice = match slice {
            Slice::Positive { offset, len } => {
                let phys_offset = nth_set_bit_extend(&mask, offset);

                let phys_len = if phys_offset >= mask.len() || len == 0 {
                    // We are past any row deletions
                    len
                } else {
                    let mask = mask.clone().sliced(phys_offset, mask.len() - phys_offset);
                    nth_set_bit_extend(&mask, len - 1).saturating_add(1)
                };

                Slice::Positive {
                    offset: phys_offset,
                    len: phys_len,
                }
            },

            // We cannot calculate the physical slice for the negative case because we don't know
            // the total length of the file.
            //
            // TODO: Bitmap-based deletion vectors (e.g. Delta, Iceberg V3) are sized as the exact
            // row-count, so the slice can be normalized to positive before reaching here.
            Slice::Negative { .. } => {
                panic!()
            },
        };

        assert!(phys_slice.len() >= slice.len());

        phys_slice
    }

    fn get_mask(&self) -> Bitmap {
        match self {
            Self::IcebergPositionDelete { mask } => {
                mask.rechunk().downcast_get(0).unwrap().values().clone()
            },
        }
    }

    pub fn len(&self) -> usize {
        match self {
            Self::IcebergPositionDelete { mask } => mask.len(),
        }
    }
}

/// Calculates the nth set bit as though `mask` were extended infinitely with trues.
fn nth_set_bit_extend(mask: &Bitmap, n: usize) -> usize {
    if let Some(n_additional) = n.checked_sub(mask.set_bits()) {
        mask.len().saturating_add(n_additional)
    } else {
        BitMask::from_bitmap(mask).nth_set_bit_idx(n, 0).unwrap()
    }
}

#[cfg(test)]
mod tests {
    use polars_utils::slice_enum::Slice;

    use super::ExternalFilterMask;

    fn split_mask<I>(mask: I, slice: Slice) -> (Vec<bool>, Slice)
    where
        I: IntoIterator<Item = bool>,
    {
        let mask = ExternalFilterMask::IcebergPositionDelete {
            mask: mask.into_iter().collect(),
        };

        let slice = mask.calc_physical_slice(slice);

        let mask = {
            let Slice::Positive { offset, len } =
                slice.clone().restrict_to_bounds(mask.get_mask().len())
            else {
                unreachable!()
            };

            mask.get_mask().sliced(offset, len)
        };

        (mask.into_iter().collect(), slice)
    }

    #[test]
    fn test_split_slice_positive() {
        const T: bool = true;
        const F: bool = false;

        {
            let mask = [];
            let slice = Slice::Positive { offset: 0, len: 0 };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(slice, Slice::Positive { offset: 0, len: 0 });
        }

        {
            let mask = [];
            let slice = Slice::Positive { offset: 1, len: 1 };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(slice, Slice::Positive { offset: 1, len: 1 });
        }

        {
            let mask = [F, T];
            let slice = Slice::Positive { offset: 0, len: 0 };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(slice, Slice::Positive { offset: 1, len: 0 });
        }

        {
            let mask = [F, T];
            let slice = Slice::Positive {
                offset: usize::MAX,
                len: 0,
            };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(
                slice,
                Slice::Positive {
                    offset: usize::MAX,
                    len: 0
                }
            );
        }

        {
            let mask = [F, T, F, F, T, F, T, F, T];
            //       true_index :     0        1     2     3  (index after deletion)
            //       phys_index :  0, 1, 2, 3, 4, 5, 6, 7, 8
            //           offset :              ^              = 4
            //              len :              ^^^^^^^        = 3 items

            let slice = Slice::Positive { offset: 1, len: 2 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F, T]);
            assert_eq!(slice, Slice::Positive { offset: 4, len: 3 });
        }

        {
            let mask = [T, F];
            let slice = Slice::Positive { offset: 0, len: 2 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F]);
            assert_eq!(slice, Slice::Positive { offset: 0, len: 3 });
        }

        {
            let mask = [T, F];
            let slice = Slice::Positive {
                offset: 100,
                len: 50,
            };
            let (mask, slice) = split_mask(mask, slice);

            assert!(mask.is_empty());
            assert_eq!(
                slice,
                Slice::Positive {
                    offset: 101,
                    len: 50
                }
            );
        }

        {
            let mask = [T, F];
            let slice = Slice::Positive { offset: 0, len: 50 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F]);
            assert_eq!(slice, Slice::Positive { offset: 0, len: 51 });
        }

        {
            let mask = [T, F, F, T, F];
            let slice = Slice::Positive { offset: 0, len: 50 };
            let (mask, slice) = split_mask(mask, slice);

            assert_eq!(mask.as_slice(), &[T, F, F, T, F]);
            assert_eq!(slice, Slice::Positive { offset: 0, len: 53 });
        }
    }
}
