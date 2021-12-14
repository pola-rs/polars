pub mod conversion;
pub mod dataframe;
pub mod dataframe_object;
pub mod datatypes;
pub mod error;
pub mod file;
pub mod lazy;
pub mod list_construction;
pub mod prelude;
pub mod series;
pub mod series_object;
use crate::dataframe::JsDataFrame;
use crate::lazy::dsl;
use crate::lazy::functions;
use crate::lazy::lazyframe_object::JsLazyFrame;
use crate::series::{repeat, JsSeries};
use napi::{JsObject, Result};

#[macro_use]
extern crate napi_derive;

#[module_exports]
pub fn init(mut exports: JsObject, env: napi::Env) -> Result<()> {
    let ldf = JsLazyFrame::to_object(&env)?;
    let series = JsSeries::to_object(&env)?;
    let df = JsDataFrame::to_object(&env)?;
    let expr = dsl::JsExpr::to_object(&env)?;
    let when = dsl::JsWhen::to_object(&env)?;
    let when_then = dsl::JsWhenThen::to_object(&env)?;
    let when_then_then = dsl::JsWhenThenThen::to_object(&env)?;

    exports.set_named_property("series", series)?;
    exports.set_named_property("df", df)?;
    exports.set_named_property("ldf", ldf)?;
    exports.set_named_property("expr", expr)?;
    exports.set_named_property("_when", when)?;
    exports.set_named_property("_whenthen", when_then)?;
    exports.set_named_property("_whenthenthen", when_then_then)?;
    exports.create_named_method("repeat", repeat)?;
    exports.create_named_method("col", dsl::col)?;
    exports.create_named_method("cols", dsl::cols)?;
    exports.create_named_method("lit", dsl::lit)?;
    exports.create_named_method("when", dsl::when)?;
    exports.create_named_method("arange", functions::arange)?;
    exports.create_named_method("argSortBy", functions::argsort_by)?;
    exports.create_named_method("concatList", functions::concat_lst)?;
    exports.create_named_method("concatString", functions::concat_str)?;
    exports.create_named_method("cov", functions::cov)?;
    exports.create_named_method("pearsonCorr", functions::pearson_corr)?;
    exports.create_named_method("spearmanRankCorr", functions::spearman_rank_corr)?;
    Ok(())
}
