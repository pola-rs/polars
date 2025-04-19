use arrow::bitmap::Bitmap;
use polars_core::frame::DataFrame;
use polars_core::prelude::{Column, IDX_DTYPE};
use polars_core::scalar::Scalar;
use polars_core::schema::Schema;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;
use polars_plan::plans::hive::HivePartitionsDf;
use polars_utils::pl_str::PlSmallStr;
use polars_utils::{IdxSize, format_pl_smallstr};

use super::MultiScanTaskInitializer;

impl MultiScanTaskInitializer {
    /// # Returns
    /// `(skip_files_mask, scan_predicate)`
    ///
    /// TODO: Move logic here, rename to `evaluate_on_constant_columns`.
    pub fn initialize_predicate(&self) -> PolarsResult<(Option<Bitmap>, Option<&ScanIOPredicate>)> {
        if let Some(predicate) = &self.config.predicate {
            if let Some(hive_parts) = self.config.hive_parts.as_ref() {
                let (skip_files_mask, need_pred_for_inner_readers) = scan_predicate_to_mask(
                    predicate,
                    self.config.projected_file_schema.as_ref(),
                    hive_parts.schema(),
                    hive_parts,
                    self.config.verbose,
                )?;

                return Ok((
                    skip_files_mask,
                    need_pred_for_inner_readers.then_some(predicate),
                ));
            }
        }

        Ok((None, self.config.predicate.as_ref()))
    }
}

fn scan_predicate_to_mask(
    scan_predicate: &ScanIOPredicate,
    file_schema: &Schema,
    hive_schema: &Schema,
    hive_parts: &HivePartitionsDf,
    verbose: bool,
) -> PolarsResult<(Option<Bitmap>, bool)> {
    let Some(sbp) = scan_predicate.skip_batch_predicate.as_ref() else {
        return Ok((None, true));
    };

    let non_hive_live_columns = scan_predicate
        .live_columns
        .iter()
        .filter(|lc| !hive_schema.contains(lc))
        .collect::<Vec<_>>();

    if non_hive_live_columns.len() == scan_predicate.live_columns.len() {
        return Ok((None, true));
    }

    let mut statistics_columns =
        Vec::with_capacity(1 + 3 * hive_schema.len() + 3 * non_hive_live_columns.len());

    // We don't know the sizes of the files here yet.
    statistics_columns.push(Column::new_scalar(
        "len".into(),
        Scalar::null(IDX_DTYPE),
        hive_parts.df().height(),
    ));
    for column in hive_parts.df().get_columns() {
        let c = column.name();

        // If the hive value is not null, we know we have 0 nulls for the hive column in the file
        // otherwise we don't know. Same reasoning as with the `len`.
        let mut nc = Column::new_scalar(
            format_pl_smallstr!("{c}_nc"),
            (0 as IdxSize).into(),
            hive_parts.df().height(),
        );
        if column.has_nulls() {
            nc = nc.zip_with_same_type(
                &column.is_null(),
                &Column::new_scalar(PlSmallStr::EMPTY, Scalar::null(IDX_DTYPE), 1),
            )?;
        }

        statistics_columns.extend([
            column.clone().with_name(format_pl_smallstr!("{c}_min")),
            column.clone().with_name(format_pl_smallstr!("{c}_max")),
            nc,
        ]);
    }
    for c in &non_hive_live_columns {
        let dtype = file_schema.try_get(c)?;
        statistics_columns.extend([
            Column::full_null(
                format_pl_smallstr!("{c}_min"),
                hive_parts.df().height(),
                dtype,
            ),
            Column::full_null(
                format_pl_smallstr!("{c}_max"),
                hive_parts.df().height(),
                dtype,
            ),
            Column::full_null(
                format_pl_smallstr!("{c}_nc"),
                hive_parts.df().height(),
                &IDX_DTYPE,
            ),
        ]);
    }

    let statistics_df = DataFrame::new(statistics_columns)?;
    let mask = sbp.evaluate_with_stat_df(&statistics_df)?;

    if verbose {
        eprintln!(
            "[MultiScan]: Predicate pushdown allows skipping {} / {} files",
            mask.set_bits(),
            mask.len()
        );
    }

    Ok((Some(mask), !non_hive_live_columns.is_empty()))
}
