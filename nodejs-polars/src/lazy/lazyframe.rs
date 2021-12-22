use crate::conversion::prelude::*;
use crate::prelude::{JsPolarsEr, JsResult};
use napi::{CallContext, JsExternal, JsObject, JsString};
use polars::lazy::frame::{LazyCsvReader, LazyFrame, LazyGroupBy};
use polars::lazy::prelude::col;
use polars::prelude::NullValues;
use polars::prelude::*;

impl IntoJs<JsExternal> for LazyFrame {
    fn try_into_js(self, cx: &CallContext) -> JsResult<JsExternal> {
        cx.env.create_external(self, None)
    }
}

impl IntoJs<JsExternal> for LazyGroupBy {
    fn try_into_js(self, cx: &CallContext) -> JsResult<JsExternal> {
        cx.env.create_external(self, None)
    }
}

#[js_function(1)]
pub fn new_from_csv(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;

    let cache: bool = params.get_or("cache", true)?;
    let comment_char: Option<&str> = params.get_as("commentChar")?;
    let has_header: bool = params.get_or("hasHeader", true)?;
    let ignore_errors: bool = params.get_or("ignoreErrors", false)?;
    let infer_schema_length: usize = params.get_or("inferSchemaLength", 100 as usize)?;
    let low_memory: bool = params.get_or("lowMemory", false)?;
    let null_values: Option<Wrap<NullValues>> = params.get_as("nullValues")?;
    let path: String = params.get_as("path")?;
    let quote_char: &str = params.get_or("quoteChar", r#"""#)?;
    let sep: &str = params.get_or("sep", ",")?;
    let skip_rows: usize = params.get_or("startRows", 0)?;
    let stop_after_n_rows: Option<usize> = params.get_as("endRows")?;

    let null_values = null_values.map(|w| w.0);
    let comment_char = comment_char.map(|s| s.as_bytes()[0]);
    let quote_char = quote_char.as_bytes()[0];
    let delimiter = sep.as_bytes()[0];

    LazyCsvReader::new(path)
        .with_infer_schema_length(Some(infer_schema_length))
        .with_delimiter(delimiter)
        .has_header(has_header)
        .with_ignore_parser_errors(ignore_errors)
        .with_skip_rows(skip_rows)
        .with_n_rows(stop_after_n_rows)
        .with_cache(cache)
        .low_memory(low_memory)
        .with_comment_char(comment_char)
        .with_quote_char(Some(quote_char))
        .with_null_values(null_values)
        .finish()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn describe_plan(cx: CallContext) -> JsResult<JsString> {
    get_params(&cx)?
        .get_external::<LazyFrame>(&cx, "_ldf")?
        .describe_plan()
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn describe_optimized_plan(cx: CallContext) -> JsResult<JsString> {
    get_params(&cx)?
        .get_external::<LazyFrame>(&cx, "_ldf")?
        .describe_optimized_plan()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn to_dot(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;
    let optimized = params.get_or::<bool>("optimized", true)?;
    ldf.to_dot(optimized)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn optimization_toggle(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;

    let type_coercion: bool = params.get_or("typeCoercion", true)?;
    let predicate_pushdown: bool = params.get_or("predicatePushdown", true)?;
    let projection_pushdown: bool = params.get_or("projectionPushdown", true)?;
    let simplify_expr: bool = params.get_or("simplifyExpr", true)?;
    let string_cache: bool = params.get_or("stringCache", true)?;

    let ldf = ldf.clone();
    ldf.with_type_coercion(type_coercion)
        .with_predicate_pushdown(predicate_pushdown)
        .with_simplify_expr(simplify_expr)
        .with_string_cache(string_cache)
        .with_projection_pushdown(projection_pushdown)
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn sort(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;
    let by_column = params.get_as::<String>("by")?;
    let reverse = params.get_or("reverse", false)?;
    ldf.clone().sort(&by_column, reverse).try_into_js(&cx)
}

#[js_function(1)]
pub fn sort_by_exprs(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;
    let by_column = params.get_external_vec::<Expr>(&cx, "by")?;
    let reverse = params.get_as::<Vec<bool>>("reverse")?;
    ldf.clone()
        .sort_by_exprs(by_column, reverse)
        .try_into_js(&cx)
}

pub struct Collect(LazyFrame);

impl napi::Task for Collect {
    type Output = DataFrame;
    type JsValue = JsExternal;

    fn compute(&mut self) -> JsResult<Self::Output> {
        self.0
            .clone()
            .collect()
            .map_err(|err| JsPolarsEr::from(err).into())
    }

    fn resolve(self, env: napi::Env, output: Self::Output) -> JsResult<Self::JsValue> {
        env.create_external(output, None)
    }
}

#[js_function(1)]
pub fn collect(cx: CallContext) -> JsResult<JsObject> {
    let ldf = get_params(&cx)?
        .get_external::<LazyFrame>(&cx, "_ldf")?
        .clone();
    let c = Collect(ldf);
    cx.env.spawn(c).map(|task| task.promise_object())
}

#[js_function(1)]
pub fn collect_sync(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<LazyFrame>(&cx, "_ldf")?
        .clone()
        .collect()
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn fetch_sync(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;
    let n_rows: usize = params.get_or("numRows", 500 as usize)?;
    ldf.clone()
        .fetch(n_rows)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn filter(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;
    let predicate = params.get_external::<Expr>(&cx, "predicate")?;
    ldf.clone().filter(predicate.clone()).try_into_js(&cx)
}

#[js_function(1)]
pub fn select(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?;
    let predicate = params.get_external_vec::<Expr>(&cx, "predicate")?;
    ldf.clone().select(predicate).try_into_js(&cx)
}

#[js_function(1)]
pub fn groupby(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let by = params.get_external_vec::<Expr>(&cx, "by")?;
    let agg = params.get_as::<String>("aggMethod")?;
    let aggs = params.get_external_vec::<Expr>(&cx, "aggs")?;
    let n = params.get_as::<Option<usize>>("n")?;
    let maintain_order: bool = params.get_or("maintainOrder", false)?;
    let lazy_gb = if maintain_order {
        ldf.groupby_stable(by)
    } else {
        ldf.groupby(by)
    };
    let ldf = match agg.as_str() {
        "head" => lazy_gb.head(n),
        "tail" => lazy_gb.tail(n),
        "agg" => lazy_gb.agg(aggs),
        a => {
            let e: JsPolarsEr =
                PolarsError::ComputeError(format!("agg fn {} does not exists", a).into()).into();
            return Err(e.into());
        }
    };
    ldf.try_into_js(&cx)
}

#[js_function(1)]
pub fn join(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let other = params.get_external::<LazyFrame>(&cx, "other")?.clone();
    let left_on = params.get_external_vec::<Expr>(&cx, "leftOn")?;
    let right_on = params.get_external_vec::<Expr>(&cx, "rightOn")?;
    let allow_parallel: bool = params.get_or("allowParallel", true)?;
    let force_parallel: bool = params.get_or("forceParallel", false)?;
    let how: String = params.get_or("how", "inner".to_owned())?;
    let suffix: String = params.get_or("suffix", "_right".to_owned())?;
    // let asof_by_left: Vec<String> = params.get_as("asofByLeft")?;
    // let asof_by_right: Vec<String> = params.get_as("asofByRight")?;
    let how = match how.as_str() {
        "left" => JoinType::Left,
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        // "asof" => JoinType::AsOf,
        // "cross" => JoinType::Cross,
        _ => panic!("not supported"),
    };
    ldf.join_builder()
        .with(other)
        .left_on(left_on)
        .right_on(right_on)
        .allow_parallel(allow_parallel)
        .force_parallel(force_parallel)
        .how(how)
        .suffix(suffix)
        // .asof_by(asof_by_left, asof_by_right)
        .finish()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn with_column(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let expr = params.get_external::<Expr>(&cx, "expr")?.clone();
    ldf.with_column(expr).try_into_js(&cx)
}

#[js_function(1)]
pub fn with_columns(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let exprs = params.get_external_vec::<Expr>(&cx, "exprs")?.clone();
    ldf.with_columns(exprs).try_into_js(&cx)
}

#[js_function(1)]
pub fn rename(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let existing: Vec<String> = params.get_as("existing")?;
    let replacements: Vec<String> = params.get_as("replacements")?;
    ldf.rename(existing, replacements).try_into_js(&cx)
}

#[js_function(1)]
pub fn with_column_renamed(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let existing: String = params.get_as("existing")?;
    let replacement: String = params.get_as("replacement")?;
    ldf.with_column_renamed(&existing, &replacement)
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn shift(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let periods: i64 = params.get_as("periods")?;
    ldf.shift(periods).try_into_js(&cx)
}

#[js_function(1)]
pub fn shift_and_fill(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let fill_value = params.get_external::<Expr>(&cx, "fillValue")?.clone();
    let periods: i64 = params.get_as("periods")?;

    ldf.shift_and_fill(periods, fill_value).try_into_js(&cx)
}

#[js_function(1)]
pub fn fill_null(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let fill_value = params.get_external::<Expr>(&cx, "fillValue")?.clone();

    ldf.fill_null(fill_value).try_into_js(&cx)
}

#[js_function(1)]
pub fn fill_nan(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let fill_value = params.get_external::<Expr>(&cx, "fillValue")?.clone();

    ldf.fill_nan(fill_value).try_into_js(&cx)
}

#[js_function(1)]
pub fn quantile(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let quantile = params.get_as::<f64>("quantile")?;

    ldf.quantile(quantile, QuantileInterpolOptions::default())
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn explode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let column = params.get_external_vec::<Expr>(&cx, "column")?.clone();

    ldf.explode(column).try_into_js(&cx)
}

#[js_function(1)]
pub fn drop_duplicates(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let maintain_order: bool = params.get_or("maintainOrder", false)?;
    let subset: Option<Vec<String>> = params.get_as("subset")?;

    ldf.drop_duplicates(maintain_order, subset).try_into_js(&cx)
}
#[js_function(1)]
pub fn drop_nulls(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let subset: Option<Vec<String>> = params.get_as("subset")?;

    ldf.drop_nulls(subset.map(|v| v.into_iter().map(|s| col(&s)).collect()))
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn slice(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let offset: i64 = params.get_as("offset")?;
    let len: usize = params.get_as("len")?;

    ldf.slice(offset, len).try_into_js(&cx)
}

#[js_function(1)]
pub fn melt(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let id_vars: Vec<String> = params.get_as("idVars")?;
    let value_vars: Vec<String> = params.get_as("valueVars")?;

    ldf.melt(id_vars, value_vars).try_into_js(&cx)
}
#[js_function(1)]
pub fn tail(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let length: usize = params.get_as("length")?;

    ldf.tail(length).try_into_js(&cx)
}

#[js_function(1)]
pub fn with_row_count(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let name: String = params.get_as("name")?;

    ldf.with_row_count(&name).try_into_js(&cx)
}
#[js_function(1)]
pub fn drop_columns(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let cols: Vec<String> = params.get_as("cols")?;

    ldf.drop_columns(cols).try_into_js(&cx)
}
#[js_function(1)]
pub fn columns(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let ldf = params.get_external::<LazyFrame>(&cx, "_ldf")?.clone();
    let columns: Vec<String> = ldf
        .schema()
        .fields()
        .iter()
        .map(|fld| fld.name().to_string())
        .collect();
    let mut arr = cx.env.create_array_with_length(columns.len())?;
    for (idx, item) in columns.into_iter().enumerate() {
        arr.set_element(idx as u32, cx.env.create_string_from_std(item)?)?;
    }
    Ok(arr)
}

macro_rules! impl_null_args_method {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            get_params(&cx)?
                .get_external::<LazyFrame>(&cx, "_ldf")?
                .clone()
                .$name()
                .try_into_js(&cx)
        }
    };
}

impl_null_args_method!(min);
impl_null_args_method!(max);
impl_null_args_method!(sum);
impl_null_args_method!(mean);
impl_null_args_method!(std);
impl_null_args_method!(var);
impl_null_args_method!(median);
impl_null_args_method!(reverse);
impl_null_args_method!(cache);
impl_null_args_method!(clone);
