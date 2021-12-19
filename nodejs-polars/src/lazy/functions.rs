use crate::conversion::prelude::*;
use crate::prelude::JsResult;
use napi::*;
use polars::lazy::functions;
use polars::prelude::*;

#[js_function(1)]
pub fn arange(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let low = params.get_external::<Expr>(&cx, "low")?.clone();
    let high = params.get_external::<Expr>(&cx, "high")?.clone();
    let step: usize = params.get_or("step", 1)?;
    let arange = functions::arange(low, high, step);
    arange.try_into_js(&cx)
}

#[js_function(1)]
pub fn argsort_by(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let by = params.get_external_vec::<Expr>(&cx, "by")?;
    let reverse = params.get_as::<Vec<bool>>("reverse")?;
    functions::argsort_by(by, &reverse).try_into_js(&cx)
}

#[js_function(1)]
pub fn concat_lst(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let items = params.get_external_vec::<Expr>(&cx, "items")?;
    functions::concat_lst(items).try_into_js(&cx)
}

#[js_function(1)]
pub fn concat_str(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let items = params.get_external_vec::<Expr>(&cx, "items")?;
    let sep: &str = params.get_or("sep", ",")?;

    functions::concat_str(items, sep).try_into_js(&cx)
}

#[js_function(1)]
pub fn cov(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let a = params.get_external::<Expr>(&cx, "a")?.clone();
    let b = params.get_external::<Expr>(&cx, "b")?.clone();
    functions::cov(a, b).try_into_js(&cx)
}

#[js_function(1)]
pub fn pearson_corr(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let a = params.get_external::<Expr>(&cx, "a")?.clone();
    let b = params.get_external::<Expr>(&cx, "b")?.clone();
    functions::pearson_corr(a, b).try_into_js(&cx)
}

#[js_function(1)]
pub fn spearman_rank_corr(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let a = params.get_external::<Expr>(&cx, "a")?.clone();
    let b = params.get_external::<Expr>(&cx, "b")?.clone();
    functions::spearman_rank_corr(a, b).try_into_js(&cx)
}
