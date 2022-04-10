use crate::conversion::prelude::*;
use crate::datatypes::JsDataType;
use crate::error::JsPolarsEr;
use crate::prelude::JsResult;
use napi::{
    CallContext, Either, JsBoolean, JsExternal, JsNumber, JsObject, JsUndefined, JsUnknown,
};
use polars::frame::groupby::GroupBy;
use polars::prelude::*;

pub struct JsDataFrame {}

impl IntoJs<JsExternal> for DataFrame {
    fn try_into_js(self, cx: &CallContext) -> JsResult<JsExternal> {
        cx.env.create_external(self, None)
    }
}


#[js_function(1)]
pub(crate) fn add(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df + s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}
#[js_function(1)]
pub(crate) fn sub(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df - s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn div(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df / s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn mul(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df * s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn rem(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "other")?;
    let df = (df % s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sample_n(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let n = params.get_as::<usize>("n")?;
    let with_replacement = params.get_as::<bool>("withReplacement")?;
    let seed = params.get_as::<Option<u64>>("seed")?;
    df.sample_n(n, with_replacement, seed)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sample_frac(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let frac = params.get_as::<f64>("frac")?;
    let with_replacement = params.get_as::<bool>("withReplacement")?;
    let seed = params.get_as::<Option<u64>>("seed")?;
    df.sample_frac(frac, with_replacement, seed)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn rechunk(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let df: DataFrame = df.agg_chunks().into();
    df.try_into_js(&cx)
}
#[js_function(1)]
pub(crate) fn as_str(cx: CallContext) -> JsResult<napi::JsBuffer> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s: String = format!("{:?}", df);
    let bytes = s.into_bytes();
    let buff_val: napi::JsBufferValue = cx.env.create_buffer_with_data(bytes)?;
    Ok(buff_val.into_raw())
}

#[js_function(1)]
pub fn fill_null(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let strategy = params.get_as::<&str>("strategy")?;
    let strat = match strategy {
        "backward" => FillNullStrategy::Backward,
        "forward" => FillNullStrategy::Forward,
        "min" => FillNullStrategy::Min,
        "max" => FillNullStrategy::Max,
        "mean" => FillNullStrategy::Mean,
        "one" => FillNullStrategy::One,
        "zero" => FillNullStrategy::Zero,
        s => return Err(JsPolarsEr::Other(format!("Strategy {} not supported", s)).into()),
    };

    df.fill_null(strat)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn join(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;
    let left_on = params.get_as::<Vec<&str>>("left_on")?;
    let right_on = params.get_as::<Vec<&str>>("right_on")?;
    let how = params.get_as::<&str>("how")?;
    let suffix = params.get_as::<String>("suffix")?;

    let how = match how {
        "left" => JoinType::Left,
        "inner" => JoinType::Inner,
        "outer" => JoinType::Outer,
        // "asof" => JoinType::AsOf,
        // "cross" => JoinType::Cross,
        _ => panic!("not supported"),
    };

    df.join(other, left_on, right_on, how, Some(suffix))
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn get_columns(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = df.get_columns().clone();
    let mut arr: JsObject = cx.env.create_array_with_length(s.len())?;

    for (idx, series) in s.into_iter().enumerate() {
        let wrapped = cx.env.create_external(series, None)?;
        arr.set_element(idx as u32, wrapped)?;
    }
    Ok(arr)
}

#[js_function(1)]
pub fn columns(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let names = df.get_column_names();
    cx.env.to_js_value(&names)
}

#[js_function(1)]
pub fn set_column_names(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let names = params.get_as::<Vec<&str>>("names")?;
    df.set_column_names(&names).map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

#[js_function(1)]
pub fn schema(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mut obj = cx.env.create_object()?;

    for (name, dtype) in df.schema().iter() {
        let field_name = format!("{}", name).try_into_js(&cx)?;
        let dtype: JsDataType = dtype.clone().into();
        let js_string = dtype.to_string().try_into_js(&cx)?;
        obj.set_property(field_name, js_string).unwrap();
    }
    Ok(obj)
}

#[js_function(1)]
pub fn with_column(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = params.get_external::<Series>(&cx, "_series")?;
    let s: Series = s.clone();
    let mut df = df.clone();
    df.with_column(s).map_err(JsPolarsEr::from)?;
    df.try_into_js(&cx)
}

#[js_function(1)]
pub fn dtypes(cx: CallContext) -> JsResult<JsUnknown> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let dtypes: Vec<String> = df
        .dtypes()
        .iter()
        .map(|arrow_dtype| {
            let dt: JsDataType = arrow_dtype.into();
            dt.to_string()
        })
        .collect();

    cx.env.to_js_value(&dtypes)
}
#[js_function(1)]
pub fn n_chunks(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let n = df.n_chunks().map_err(JsPolarsEr::from)?;
    cx.env.create_int64(n as i64)
}

#[js_function(1)]
pub fn shape(cx: CallContext) -> JsResult<JsObject> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let (height, width) = df.shape();
    let height = height.into_js(&cx);
    let width = width.into_js(&cx);
    let mut obj = cx.env.create_object()?;
    obj.set_named_property("height", height)?;
    obj.set_named_property("width", width)?;

    Ok(obj)
}

#[js_function(1)]
pub fn height(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.height().try_into_js(&cx)
}

#[js_function(1)]
pub fn width(cx: CallContext) -> JsResult<JsNumber> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.width().try_into_js(&cx)
}

#[js_function(1)]
pub fn hstack(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let column_obj: JsObject = params.get::<JsObject>("columns")?;
    let in_place = params.get_as::<bool>("in_place")?;

    let len = column_obj.get_array_length()?;
    let mut columns: Vec<Series> = Vec::with_capacity(len as usize);

    for idx in 0..len {
        let item: JsExternal = column_obj.get_element(idx)?;
        let s: &Series = cx.env.get_value_external(&item)?;
        columns.push(s.clone())
    }

    if in_place {
        df.hstack_mut(&columns).map_err(JsPolarsEr::from)?;
        cx.env.get_undefined().map(Either::B)
    } else {
        let df = df.hstack(&columns).map_err(JsPolarsEr::from)?;
        df.try_into_js(&cx).map(Either::A)
    }
}

#[js_function(1)]
pub fn vstack(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;

    df.vstack(other).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn drop_in_place(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    df.drop_in_place(name)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn drop_nulls(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let subset = params.get_as::<Option<Vec<String>>>("subset")?;
    df.drop_nulls(subset.as_ref().map(|s| s.as_ref()))
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn drop(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    df.drop(name).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}

#[js_function(1)]
pub fn select_at_idx(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<usize>("index")?;

    let opt = df.select_at_idx(idx).map(|s| s.clone());

    match opt {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn find_idx_by_name(cx: CallContext) -> JsResult<Either<JsNumber, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;
    let opt = df.find_idx_by_name(name);

    match opt {
        Some(idx) => idx.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn column(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<&str>("name")?;

    df.column(name)
        .map(|s| s.clone())
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn select(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let selection = params.get_as::<Vec<&str>>("selection")?;
    df.select(&selection)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn filter(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mask = params.get_external::<Series>(&cx, "mask")?;
    if let Ok(ca) = mask.bool() {
        df.filter(ca).map_err(JsPolarsEr::from)?.try_into_js(&cx)
    } else {
        Err(napi::Error::from_reason(
            "Expected a boolean mask".to_owned(),
        ))
    }
}
#[js_function(1)]
pub fn take(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let indices = params.get::<JsObject>("indices")?;
    let len = indices.get_array_length()?;
    let indices: Vec<u32> = (0..len)
        .map(|v| {
            let wv: WrappedValue = indices
                .get_element_unchecked::<JsUnknown>(v)
                .unwrap()
                .into();
            wv.extract::<u32>().unwrap()
        })
        .collect();

    let indices = UInt32Chunked::from_vec("", indices);
    df.take(&indices)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn take_with_series(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let indices = params.get_external::<Series>(&cx, "indices")?;
    let idx = indices.u32().map_err(JsPolarsEr::from)?;

    df.take(idx).map_err(JsPolarsEr::from)?.try_into_js(&cx)
}
#[js_function(1)]
pub fn sort(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let by_columns = params.get_as::<Vec<String>>("by")?;
    let reverse = params.get_as::<bool>("reverse")?;
    df.sort(by_columns, reverse)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn sort_in_place(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let by_column = params.get_as::<&str>("by")?;
    let reverse = params.get_as::<bool>("reverse")?;

    df.sort_in_place([by_column], reverse)
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub fn replace(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;

    let column = params.get_as::<&str>("column")?;
    let new_col = params.get_external::<Series>(&cx, "new_col")?;

    df.replace(column, new_col.clone())
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}

#[js_function(1)]
pub fn rename(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;

    let column = params.get_as::<&str>("column")?;
    let new_col = params.get_as::<&str>("new_col")?;

    df.rename(column, new_col).map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}
#[js_function(1)]
pub fn replace_at_idx(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<usize>("index")?;
    let new_col = params.get_external::<Series>(&cx, "newColumn")?;
    df.replace_at_idx(idx, new_col.clone())
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub fn insert_at_idx(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let idx = params.get_as::<usize>("index")?;
    let new_col = params.get_external::<Series>(&cx, "new_col")?;

    df.insert_at_idx(idx, new_col.clone())
        .map_err(JsPolarsEr::from)?;

    cx.env.get_undefined()
}
#[js_function(1)]
pub fn slice(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let offset = params.get_as::<usize>("offset")?;
    let length = params.get_as::<usize>("length")?;

    df.slice(offset as i64, length).try_into_js(&cx)
}
#[js_function(1)]
pub fn head(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let length = params.get_as::<Option<usize>>("length")?;
    df.head(length).try_into_js(&cx)
}
#[js_function(1)]
pub fn tail(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let length = params.get_as::<Option<usize>>("length")?;
    df.tail(length).try_into_js(&cx)
}
#[js_function(1)]
pub fn is_unique(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mask = df.is_unique().map_err(JsPolarsEr::from)?;
    mask.into_series().try_into_js(&cx)
}
#[js_function(1)]
pub fn is_duplicated(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let mask = df.is_duplicated().map_err(JsPolarsEr::from)?;
    mask.into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn frame_equal(cx: CallContext) -> JsResult<JsBoolean> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;
    let null_equal = params.get_as::<bool>("nullEqual")?;
    let eq = if null_equal {
        df.frame_equal_missing(other)
    } else {
        df.frame_equal(other)
    };

    cx.env.get_boolean(eq)
}
#[js_function(1)]
pub fn with_row_count(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let name = params.get_as::<String>("name")?;
    let offset: Option<u32> = params.get_as("offset")?;

    df.with_row_count(&name, offset)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn groupby(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let by = params.get_as::<Vec<&str>>("by")?;
    let agg = params.get_as::<&str>("agg")?;
    let select = params.get_as::<Option<Vec<String>>>("select")?;
    let gb = df.groupby(&by).map_err(JsPolarsEr::from)?;

    let selection = match select.as_ref() {
        Some(s) => gb.select(s),
        None => gb,
    };
    finish_groupby(selection, agg)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn clone(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.clone().try_into_js(&cx)
}
#[js_function(1)]
pub fn melt(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let id_vars = params.get_as::<Vec<&str>>("idVars")?;
    let value_vars = params.get_as::<Vec<&str>>("valueVars")?;

    df.melt(id_vars, value_vars)
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn shift(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let periods = params.get_as::<i64>("periods")?;

    df.shift(periods).try_into_js(&cx)
}

#[js_function(1)]
pub fn extend(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let other = params.get_external::<DataFrame>(&cx, "other")?;

    df.extend(other).map_err(JsPolarsEr::from)?;
    cx.env.get_undefined()
}

#[js_function(1)]
pub fn unique(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let maintain_order: bool = params.get_or("maintainOrder", false)?;
    let keep: UniqueKeepStrategy = params.get_as("keep")?;

    let subset: Option<Vec<String>> = params.get_as("subset")?;
    let subset = subset.as_ref().map(|v| v.as_ref());
    let df = match maintain_order {
        true => df.unique_stable(subset, keep),
        false => df.unique(subset, keep),
    }
    .map_err(JsPolarsEr::from)?;

    df.try_into_js(&cx)
}

#[js_function(1)]
pub fn max(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.max().try_into_js(&cx)
}
#[js_function(1)]
pub fn min(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.min().try_into_js(&cx)
}
#[js_function(1)]
pub fn sum(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.sum().try_into_js(&cx)
}
#[js_function(1)]
pub fn mean(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.mean().try_into_js(&cx)
}
#[js_function(1)]
pub fn std(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.std().try_into_js(&cx)
}
#[js_function(1)]
pub fn var(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.var().try_into_js(&cx)
}
#[js_function(1)]
pub fn median(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.median().try_into_js(&cx)
}
#[js_function(1)]
pub fn null_count(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.null_count().try_into_js(&cx)
}

#[js_function(1)]
pub fn hmax(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = df.hmax().map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn hmean(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let strategy = params.get_as::<&str>("nullStrategy")?;
    let strategy = str_to_null_strategy(strategy)?;

    let s = df.hmean(strategy).map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn hmin(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let s = df.hmin().map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn hsum(cx: CallContext) -> JsResult<Either<JsExternal, JsUndefined>> {
    let params = get_params(&cx)?;
    let strategy = params.get_as::<&str>("nullStrategy")?;
    let strategy = str_to_null_strategy(strategy)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;

    let s = df.hsum(strategy).map_err(JsPolarsEr::from)?;
    match s {
        Some(s) => s.try_into_js(&cx).map(Either::A),
        None => cx.env.get_undefined().map(Either::B),
    }
}

#[js_function(1)]
pub fn quantile(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let quantile = params.get_as::<f64>("quantile")?;
    df.quantile(quantile, QuantileInterpolOptions::default())
        .map_err(JsPolarsEr::from)?
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn shrink_to_fit(cx: CallContext) -> JsResult<JsUndefined> {
    let params = get_params(&cx)?;
    let df = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    df.shrink_to_fit();
    cx.env.get_undefined()
}

#[js_function(1)]
pub fn hash_rows(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    let k0 = params.get_as::<u64>("k0")?;
    let k1 = params.get_as::<u64>("k1")?;
    let k2 = params.get_as::<u64>("k2")?;
    let k3 = params.get_as::<u64>("k3")?;

    let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
    let hash = df.hash_rows(Some(hb)).map_err(JsPolarsEr::from)?;
    hash.into_series().try_into_js(&cx)
}

#[js_function(1)]
pub fn transpose(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df: &mut DataFrame = params.get_external_mut::<DataFrame>(&cx, "_df")?;
    let include_header: bool = params.get_or("includeHeader", false)?;
    let mut new_df = df.transpose().map_err(JsPolarsEr::from)?;
    if include_header {
        let name: String = params.get_or("headerName", "column".to_owned())?;
        let column_names = df.get_columns().iter().map(|s| s.name());
        let s = Utf8Chunked::from_iter_values(&name, column_names).into_series();
        new_df.insert_at_idx(0, s).unwrap();
    }
    new_df.try_into_js(&cx)
}

#[js_function(1)]
pub fn lazy(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let df = params.get_external::<DataFrame>(&cx, "_df")?;
    df.clone().lazy().try_into_js(&cx)
}

fn finish_groupby(gb: GroupBy, agg: &str) -> Result<DataFrame> {
    match agg {
        "min" => gb.min(),
        "max" => gb.max(),
        "mean" => gb.mean(),
        "first" => gb.first(),
        "last" => gb.last(),
        "sum" => gb.sum(),
        "count" => gb.count(),
        "n_unique" => gb.n_unique(),
        "median" => gb.median(),
        "agg_list" => gb.agg_list(),
        "groups" => gb.groups(),
        "std" => gb.std(),
        "var" => gb.var(),
        a => Err(PolarsError::ComputeError(
            format!("agg fn {} does not exists", a).into(),
        )),
    }
}
