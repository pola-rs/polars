use crate::lazy::lazyframe;
use crate::prelude::JsResult;
use napi::JsObject;

pub struct JsLazyFrame {}

impl JsLazyFrame {
    pub fn to_object(env: &napi::Env) -> JsResult<JsObject> {
        let mut ldf = env.create_object()?;
        ldf.define_properties(&[
            napi::Property::new(env, "scanCSV")?.with_method(lazyframe::scan_csv),
            napi::Property::new(env, "scanParquet")?.with_method(lazyframe::scan_parquet),
            napi::Property::new(env, "scanIPC")?.with_method(lazyframe::scan_ipc),
            napi::Property::new(env, "describePlan")?.with_method(lazyframe::describe_plan),
            napi::Property::new(env, "describeOptimizedPlan")?
                .with_method(lazyframe::describe_optimized_plan),
            napi::Property::new(env, "cache")?.with_method(lazyframe::cache),
            napi::Property::new(env, "clone")?.with_method(lazyframe::clone),
            napi::Property::new(env, "collect")?.with_method(lazyframe::collect),
            napi::Property::new(env, "collectSync")?.with_method(lazyframe::collect_sync),
            napi::Property::new(env, "columns")?.with_method(lazyframe::columns),
            napi::Property::new(env, "dropColumns")?.with_method(lazyframe::drop_columns),
            napi::Property::new(env, "unique")?.with_method(lazyframe::unique),
            napi::Property::new(env, "dropNulls")?.with_method(lazyframe::drop_nulls),
            napi::Property::new(env, "explode")?.with_method(lazyframe::explode),
            napi::Property::new(env, "fetch")?.with_method(lazyframe::fetch),
            napi::Property::new(env, "fetchSync")?.with_method(lazyframe::fetch_sync),
            napi::Property::new(env, "fillNan")?.with_method(lazyframe::fill_nan),
            napi::Property::new(env, "fillNull")?.with_method(lazyframe::fill_null),
            napi::Property::new(env, "filter")?.with_method(lazyframe::filter),
            napi::Property::new(env, "groupby")?.with_method(lazyframe::groupby),
            napi::Property::new(env, "join")?.with_method(lazyframe::join),
            napi::Property::new(env, "max")?.with_method(lazyframe::max),
            napi::Property::new(env, "mean")?.with_method(lazyframe::mean),
            napi::Property::new(env, "median")?.with_method(lazyframe::median),
            napi::Property::new(env, "melt")?.with_method(lazyframe::melt),
            napi::Property::new(env, "min")?.with_method(lazyframe::min),
            napi::Property::new(env, "optimizationToggle")?
                .with_method(lazyframe::optimization_toggle),
            napi::Property::new(env, "quantile")?.with_method(lazyframe::quantile),
            napi::Property::new(env, "rename")?.with_method(lazyframe::rename),
            napi::Property::new(env, "reverse")?.with_method(lazyframe::reverse),
            napi::Property::new(env, "select")?.with_method(lazyframe::select),
            napi::Property::new(env, "shift")?.with_method(lazyframe::shift),
            napi::Property::new(env, "shiftAndFill")?.with_method(lazyframe::shift_and_fill),
            napi::Property::new(env, "slice")?.with_method(lazyframe::slice),
            napi::Property::new(env, "sort_by_exprs")?.with_method(lazyframe::sort_by_exprs),
            napi::Property::new(env, "sort")?.with_method(lazyframe::sort),
            napi::Property::new(env, "std")?.with_method(lazyframe::std),
            napi::Property::new(env, "sum")?.with_method(lazyframe::sum),
            napi::Property::new(env, "tail")?.with_method(lazyframe::tail),
            napi::Property::new(env, "toDot")?.with_method(lazyframe::to_dot),
            napi::Property::new(env, "var")?.with_method(lazyframe::var),
            napi::Property::new(env, "withColumn")?.with_method(lazyframe::with_column),
            napi::Property::new(env, "withColumnRenamed")?
                .with_method(lazyframe::with_column_renamed),
            napi::Property::new(env, "withColumns")?.with_method(lazyframe::with_columns),
            napi::Property::new(env, "withRowCount")?.with_method(lazyframe::with_row_count),
        ])?;
        Ok(ldf)
    }
}
