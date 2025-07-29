use std::ops::Range;

use arrow::array::{MutablePrimitiveArray, PrimitiveArray};
use arrow::bitmap::Bitmap;
use arrow::pushable::Pushable;
use polars_core::prelude::*;
use polars_io::RowIndex;
use polars_io::predicates::ScanIOPredicate;
use polars_io::prelude::FileMetadata;
use polars_parquet::read::RowGroupMetadata;
use polars_parquet::read::statistics::{ArrowColumnStatisticsArrays, deserialize_all};
use polars_utils::format_pl_smallstr;

use crate::async_executor::{self, TaskPriority};
use crate::nodes::io_sources::parquet::projection::ArrowFieldProjection;

struct StatisticsColumns {
    min: Column,
    max: Column,
    null_count: Column,
}

impl StatisticsColumns {
    fn new_null(dtype: &DataType, height: usize) -> Self {
        Self {
            min: Column::full_null(PlSmallStr::EMPTY, height, dtype),
            max: Column::full_null(PlSmallStr::EMPTY, height, dtype),
            null_count: Column::full_null(PlSmallStr::EMPTY, height, &IDX_DTYPE),
        }
    }

    fn from_arrow_statistics(
        statistics: ArrowColumnStatisticsArrays,
        field: &ArrowField,
    ) -> PolarsResult<Self> {
        Ok(Self {
            min: unsafe {
                Series::_try_from_arrow_unchecked_with_md(
                    PlSmallStr::EMPTY,
                    vec![statistics.min_value],
                    field.dtype(),
                    field.metadata.as_deref(),
                )
            }?
            .into_column(),

            max: unsafe {
                Series::_try_from_arrow_unchecked_with_md(
                    PlSmallStr::EMPTY,
                    vec![statistics.max_value],
                    field.dtype(),
                    field.metadata.as_deref(),
                )
            }?
            .into_column(),

            null_count: Series::from_arrow(PlSmallStr::EMPTY, statistics.null_count.boxed())?
                .into_column(),
        })
    }

    fn with_base_column_name(self, base_column_name: &str) -> Self {
        let b = base_column_name;

        let min = self.min.with_name(format_pl_smallstr!("{b}_min"));
        let max = self.max.with_name(format_pl_smallstr!("{b}_max"));
        let null_count = self.null_count.with_name(format_pl_smallstr!("{b}_nc"));

        Self {
            min,
            max,
            null_count,
        }
    }
}

pub(super) async fn calculate_row_group_pred_pushdown_skip_mask(
    row_group_slice: Range<usize>,
    use_statistics: bool,
    predicate: Option<&ScanIOPredicate>,
    metadata: &Arc<FileMetadata>,
    projected_arrow_fields: Arc<[ArrowFieldProjection]>,
    // This is mut so that the offset is updated to the position of the first
    // row group.
    mut row_index: Option<RowIndex>,
    verbose: bool,
) -> PolarsResult<Option<Bitmap>> {
    if !use_statistics {
        return Ok(None);
    }

    let Some(predicate) = predicate else {
        return Ok(None);
    };

    let Some(sbp) = predicate.skip_batch_predicate.as_ref() else {
        return Ok(None);
    };

    let sbp = sbp.clone();

    let num_row_groups = row_group_slice.len();
    let metadata = metadata.clone();
    let live_columns = predicate.live_columns.clone();

    // Note: We are spawning here onto the computational async runtime because the caller is being run
    // on a tokio async thread.
    let skip_row_group_mask = async_executor::spawn(TaskPriority::High, async move {
        let row_groups_slice = &metadata.row_groups[row_group_slice.clone()];

        if let Some(ri) = &mut row_index {
            for md in metadata.row_groups[0..row_group_slice.start].iter() {
                ri.offset = ri
                    .offset
                    .saturating_add(IdxSize::try_from(md.num_rows()).unwrap_or(IdxSize::MAX));
            }
        }

        let mut columns = Vec::with_capacity(1 + live_columns.len() * 3);

        let lengths: Vec<IdxSize> = row_groups_slice
            .iter()
            .map(|rg| rg.num_rows() as IdxSize)
            .collect();

        columns.push(Column::new("len".into(), lengths));

        for projection in projected_arrow_fields.iter() {
            let c = projection.output_name();

            if !live_columns.contains(c) {
                continue;
            }

            let mut statistics = load_parquet_column_statistics(row_groups_slice, projection)?;

            // Note: Order is important here. We re-use the transform for the output column, meaning
            // that it may set the column name.
            statistics.min = projection.apply_transform(statistics.min)?;
            statistics.max = projection.apply_transform(statistics.max)?;

            let statistics = statistics.with_base_column_name(c);

            columns.extend([statistics.min, statistics.max, statistics.null_count]);
        }

        if let Some(row_index) = row_index {
            let statistics = build_row_index_statistics(&row_index, row_groups_slice)
                .with_base_column_name(&row_index.name);

            columns.extend([statistics.min, statistics.max, statistics.null_count]);
        }

        let statistics_df = DataFrame::new_with_height(num_row_groups, columns)?;

        sbp.evaluate_with_stat_df(&statistics_df)
    })
    .await?;

    if verbose {
        eprintln!(
            "[ParquetFileReader]: Predicate pushdown: \
            reading {} / {} row groups",
            skip_row_group_mask.unset_bits(),
            num_row_groups,
        );
    }

    Ok(Some(skip_row_group_mask))
}

fn load_parquet_column_statistics(
    row_groups: &[RowGroupMetadata],
    projection: &ArrowFieldProjection,
) -> PolarsResult<StatisticsColumns> {
    let arrow_field = projection.arrow_field();

    let null_statistics = || {
        Ok(StatisticsColumns::new_null(
            &DataType::from_arrow_field(arrow_field),
            row_groups.len(),
        ))
    };

    // This can be None in the allow_missing_columns case.
    let Some(idxs) = row_groups[0].columns_idxs_under_root_iter(&arrow_field.name) else {
        return null_statistics();
    };

    // 0 is possible for possible for empty structs.
    //
    // 2+ is for structs. We don't support reading nested statistics for now. It does not
    // really make any sense at the moment with how we structure statistics.
    if idxs.is_empty() || idxs.len() > 1 {
        return null_statistics();
    }

    let idx = idxs[0];

    let Some(statistics) = deserialize_all(arrow_field, row_groups, idx)? else {
        return null_statistics();
    };

    StatisticsColumns::from_arrow_statistics(statistics, arrow_field)
}

fn build_row_index_statistics(
    row_index: &RowIndex,
    row_groups: &[RowGroupMetadata],
) -> StatisticsColumns {
    let mut offset = row_index.offset;

    let null_count = PrimitiveArray::<IdxSize>::full(row_groups.len(), 0, ArrowDataType::IDX_DTYPE);

    let mut min_value = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());
    let mut max_value = MutablePrimitiveArray::<IdxSize>::with_capacity(row_groups.len());

    for rg in row_groups.iter() {
        let n_rows = IdxSize::try_from(rg.num_rows()).unwrap_or(IdxSize::MAX);

        if offset.checked_add(n_rows).is_none() {
            min_value.push_null();
            max_value.push_null();
            continue;
        }

        if n_rows == 0 {
            min_value.push_null();
            max_value.push_null();
        } else {
            min_value.push_value(offset);
            max_value.push_value(offset + n_rows - 1);
        }

        offset = offset.saturating_add(n_rows);
    }

    StatisticsColumns {
        min: Series::from_array(PlSmallStr::EMPTY, min_value.freeze()).into_column(),
        max: Series::from_array(PlSmallStr::EMPTY, max_value.freeze()).into_column(),
        null_count: Series::from_array(PlSmallStr::EMPTY, null_count).into_column(),
    }
}
