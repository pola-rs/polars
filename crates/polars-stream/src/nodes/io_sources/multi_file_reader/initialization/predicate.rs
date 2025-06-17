use arrow::bitmap::Bitmap;
use polars_error::PolarsResult;
use polars_io::predicates::ScanIOPredicate;

use super::MultiScanTaskInitializer;

impl MultiScanTaskInitializer {
    /// # Returns
    /// `(skip_files_mask, scan_predicate)`
    ///
    /// TODO: Move logic here, rename to `evaluate_on_constant_columns`.
    pub fn initialize_predicate(&self) -> PolarsResult<(Option<Bitmap>, Option<&ScanIOPredicate>)> {
        if let Some(predicate) = &self.config.predicate {
            if let Some(hive_parts) = self.config.hive_parts.as_ref() {
                let mut skip_files_mask = None;

                if let Some(predicate) = &predicate.hive_predicate {
                    let mask = predicate
                        .evaluate_io(hive_parts.df())?
                        .bool()?
                        .rechunk()
                        .into_owned()
                        .downcast_into_iter()
                        .next()
                        .unwrap()
                        .values()
                        .clone();

                    // TODO: Optimize to avoid doing this
                    let mask = !&mask;

                    if self.config.verbose {
                        eprintln!(
                            "[MultiScan]: Predicate pushdown allows skipping {} / {} files",
                            mask.set_bits(),
                            mask.len()
                        );
                    }

                    skip_files_mask = Some(mask);
                }

                let need_pred_for_inner_readers = !predicate.hive_predicate_is_full_predicate;

                return Ok((
                    skip_files_mask,
                    need_pred_for_inner_readers.then_some(predicate),
                ));
            }
        }

        Ok((None, self.config.predicate.as_ref()))
    }
}
