use arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;

use super::MultiScanTaskInitializer;

impl MultiScanTaskInitializer {
    /// # Returns
    /// `(skip_files_mask, scan_predicate)`
    pub fn initialize_predicate(&self) -> PolarsResult<(Option<Bitmap>, Option<&ScanIOPredicate>)> {
        if let Some(predicate) = &self.config.predicate {
            if let Some(hive_parts) = self.config.hive_parts.as_ref() {
                let (skip_files_mask, need_pred_for_inner_readers) =
                    crate::nodes::io_sources::multi_scan::scan_predicate_to_mask(
                        predicate,
                        self.config.projected_file_schema.as_ref(),
                        hive_parts.schema(),
                        hive_parts,
                    )?;

                if need_pred_for_inner_readers {
                    return Ok((skip_files_mask, Some(predicate)));
                }
            }
        }

        Ok((None, None))
    }
}
