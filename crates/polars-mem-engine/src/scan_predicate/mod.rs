pub mod functions;
pub mod skip_files_mask;
#[cfg(feature = "dtype-categorical")]
mod table_statistics;
use core::fmt;
use std::sync::Arc;

pub use functions::{create_scan_predicate, initialize_scan_predicate};
use polars_arrow::bitmap::Bitmap;
use polars_buffer::Buffer;
use polars_core::frame::DataFrame;
use polars_core::prelude::{PlIndexMap, PlIndexSet};
use polars_core::schema::SchemaRef;
use polars_error::PolarsResult;
use polars_expr::prelude::{PhysicalExpr, phys_expr_to_io_expr};
use polars_io::predicates::{
    ColumnPredicate, DynamicColumnPredicate, DynamicPredicateSource, RuntimeRangeHint,
    ScanIOPredicate, SkipBatchPredicate, SpecializedColumnPredicate, StagedScanIOPredicate,
};
use polars_plan::plans::expr_ir::ExprIR;
use polars_utils::pl_str::PlSmallStr;

/// [`ScanPredicate::predicate`] split per column, see [`StagedScanIOPredicate`].
#[derive(Clone)]
pub struct StagedScanPredicate {
    pub column_predicates: PlIndexMap<PlSmallStr, PhysicalColumnPredicate>,
    pub rest: Option<Arc<dyn PhysicalExpr>>,
}

#[derive(Clone)]
pub struct PhysicalColumnPredicate {
    pub predicate: Option<Arc<dyn PhysicalExpr>>,
    pub specialized: Option<SpecializedColumnPredicate>,
    pub dynamic: Vec<(Arc<dyn PhysicalExpr>, Arc<dyn DynamicPredicateSource>)>,
}

impl StagedScanPredicate {
    fn to_io(&self) -> StagedScanIOPredicate {
        StagedScanIOPredicate {
            column_predicates: Arc::new(
                self.column_predicates
                    .iter()
                    .map(|(name, p)| {
                        (
                            name.clone(),
                            ColumnPredicate {
                                predicate: p.predicate.clone().map(phys_expr_to_io_expr),
                                specialized: p.specialized.clone(),
                                dynamic: p
                                    .dynamic
                                    .iter()
                                    .map(|(predicate, source)| DynamicColumnPredicate {
                                        predicate: phys_expr_to_io_expr(predicate.clone()),
                                        source: source.clone(),
                                    })
                                    .collect(),
                            },
                        )
                    })
                    .collect(),
            ),
            rest: self.rest.clone().map(phys_expr_to_io_expr),
        }
    }
}

/// All the expressions and metadata used to filter out rows using predicates.
#[derive(Clone)]
pub struct ScanPredicate {
    pub predicate: Arc<dyn PhysicalExpr>,

    /// `predicate` split for readers that filter while decoding.
    pub predicate_minterm_eirs: Option<Buffer<ExprIR>>,

    pub staged: Option<StagedScanPredicate>,

    /// Whether `predicate` filters rows at all. False when the scan only has
    /// runtime ranges to skip batches by.
    pub filters_rows: bool,

    /// Column names that are used in the predicate.
    pub live_columns: Arc<PlIndexSet<PlSmallStr>>,

    /// A predicate expression used to skip record batches based on its statistics.
    ///
    /// This expression will be given a batch size along with a `min`, `max` and `null count` for
    /// each live column (set to `null` when it is not known) and the expression evaluates to
    /// `true` if the whole batch can for sure be skipped. This may be conservative and evaluate to
    /// `false` even when the batch could theoretically be skipped.
    pub skip_batch_predicate: Option<Arc<dyn PhysicalExpr>>,

    /// Columns whose batches are skipped by a range published at run time.
    pub runtime_ranges: Vec<RuntimeRangeHint>,

    /// Predicate only referring to hive columns.
    pub hive_predicate: Option<Arc<dyn PhysicalExpr>>,
    pub hive_predicate_is_full_predicate: bool,
}

impl fmt::Debug for ScanPredicate {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        f.write_str("scan_predicate")
    }
}

/// Helper to implement [`SkipBatchPredicate`].
struct SkipBatchPredicateHelper {
    skip_batch_predicate: Arc<dyn PhysicalExpr>,
    schema: SchemaRef,
}

impl ScanPredicate {
    /// Create a predicate to skip batches using statistics.
    pub(crate) fn to_dyn_skip_batch_predicate(
        &self,
        schema: SchemaRef,
    ) -> Option<Arc<dyn SkipBatchPredicate>> {
        let skip_batch_predicate = self.skip_batch_predicate.as_ref()?.clone();
        Some(Arc::new(SkipBatchPredicateHelper {
            skip_batch_predicate,
            schema,
        }))
    }

    pub fn to_io(
        &self,
        skip_batch_predicate: Option<&Arc<dyn SkipBatchPredicate>>,
        schema: SchemaRef,
    ) -> (ScanIOPredicate, Option<Buffer<ExprIR>>) {
        (
            ScanIOPredicate {
                predicate: phys_expr_to_io_expr(self.predicate.clone()),
                staged: self.staged.as_ref().map(StagedScanPredicate::to_io),
                filters_rows: self.filters_rows,
                live_columns: self.live_columns.clone(),
                skip_batch_predicate: skip_batch_predicate
                    .cloned()
                    .or_else(|| self.to_dyn_skip_batch_predicate(schema)),
                runtime_ranges: self.runtime_ranges.clone(),
                hive_predicate: self.hive_predicate.clone().map(phys_expr_to_io_expr),
                hive_predicate_is_full_predicate: self.hive_predicate_is_full_predicate,
            },
            self.predicate_minterm_eirs.clone(),
        )
    }
}

impl SkipBatchPredicate for SkipBatchPredicateHelper {
    fn schema(&self) -> &SchemaRef {
        &self.schema
    }

    fn evaluate_with_stat_df(&self, df: &DataFrame) -> PolarsResult<Bitmap> {
        if df.height() == 0 {
            return Ok(Bitmap::new());
        }
        let array = self
            .skip_batch_predicate
            .evaluate(df, &Default::default())?;
        let array = array.bool()?.rechunk();
        let array = array.downcast_as_array();

        let array = if let Some(validity) = array.validity() {
            array.values() & validity
        } else {
            array.values().clone()
        };

        // @NOTE: Certain predicates like `1 == 1` will only output 1 value. We need to broadcast
        // the result back to the dataframe length.
        if array.len() == 1 && df.height() != 0 {
            return Ok(Bitmap::new_with_value(array.get_bit(0), df.height()));
        }

        assert_eq!(array.len(), df.height());
        Ok(array)
    }
}
