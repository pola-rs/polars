use polars_core::prelude::*;

pub trait PhysicalIoExpr: Send + Sync {
    /// Take a `DataFrame` and produces a boolean `Series` that serves
    /// as a predicate mask
    fn evaluate(&self, df: &DataFrame) -> PolarsResult<Series>;

    /// Can take &dyn Statistics and determine of a file should be
    /// read -> `true`
    /// or not -> `false`
    #[cfg(feature = "parquet")]
    fn as_stats_evaluator(&self) -> Option<&dyn StatsEvaluator> {
        None
    }
}

#[cfg(feature = "parquet")]
pub trait StatsEvaluator {
    fn should_read(&self, stats: &crate::parquet::predicates::BatchStats) -> PolarsResult<bool>;
}

#[cfg(feature = "parquet")]
pub(crate) fn arrow_schema_to_empty_df(schema: &ArrowSchema) -> DataFrame {
    let columns = schema
        .fields
        .iter()
        .map(|fld| Series::full_null(&fld.name, 0, &fld.data_type().into()))
        .collect();
    DataFrame::new_no_checks(columns)
}

#[cfg(any(feature = "parquet", feature = "json",))]
pub(crate) fn apply_predicate(
    df: &mut DataFrame,
    predicate: Option<&dyn PhysicalIoExpr>,
    parallel: bool,
) -> PolarsResult<()> {
    if let (Some(predicate), false) = (&predicate, df.is_empty()) {
        let s = predicate.evaluate(df)?;
        let mask = s.bool().expect("filter predicates was not of type boolean");

        if parallel {
            *df = df.filter(mask)?;
        } else {
            *df = df._filter_seq(mask)?;
        }
    }
    Ok(())
}
