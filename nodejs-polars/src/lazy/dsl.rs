use crate::conversion::prelude::*;
use crate::conversion::utils;
use crate::datatypes::JsDataType;
use crate::prelude::JsResult;
use napi::*;
use polars::lazy::dsl::Operator;

use crate::error::JsPolarsEr;
use polars::lazy::dsl;
use polars::prelude::*;

pub struct JsExpr {}

pub struct JsWhen {}

pub struct JsWhenThen {}

pub struct JsWhenThenThen {}

impl IntoJs<JsExternal> for Expr {
    fn try_into_js(self, cx: &CallContext) -> JsResult<JsExternal> {
        cx.env.create_external(self, None)
    }
}

#[js_function(1)]
pub(crate) fn add(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    let other = params.get_external::<Expr>(&cx, "other")?.clone();
    dsl::binary_expr(expr, Operator::Plus, other).try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn sub(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    let other = params.get_external::<Expr>(&cx, "other")?.clone();
    dsl::binary_expr(expr, Operator::Minus, other).try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn mul(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    let other = params.get_external::<Expr>(&cx, "other")?.clone();
    dsl::binary_expr(expr, Operator::Multiply, other).try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn div(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    let other = params.get_external::<Expr>(&cx, "other")?.clone();
    dsl::binary_expr(expr, Operator::Divide, other).try_into_js(&cx)
}

#[js_function(1)]
pub(crate) fn rem(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    let other = params.get_external::<Expr>(&cx, "other")?.clone();
    dsl::binary_expr(expr, Operator::Modulus, other).try_into_js(&cx)
}

#[js_function(1)]
pub fn col(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let n = params.get_as::<&str>("name")?;
    dsl::col(n).try_into_js(&cx)
}

#[js_function(1)]
pub fn cols(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let names = params.get_as::<Vec<String>>("name")?;
    dsl::cols(names).try_into_js(&cx)
}

#[js_function(1)]
pub fn lit(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let value = params.get::<JsUnknown>("value")?;
    let lit = match value.get_type()? {
        ValueType::Boolean => bool::from_js(value).map(dsl::lit)?,
        ValueType::Bigint => u64::from_js(value).map(dsl::lit)?,
        ValueType::Number => {
            let f_val = f64::from_js(value)?;
            if f_val.round() == f_val {
                let val = f_val as i64;
                if val > 0 && val < i32::MAX as i64 || val < 0 && val > i32::MIN as i64 {
                    dsl::lit(val as i32)
                } else {
                    dsl::lit(val)
                }
            } else {
                dsl::lit(f_val)
            }
        }
        ValueType::Object => {
            let obj: JsObject = unsafe { value.cast() };
            if obj.is_date()? {
                let d: JsDate = unsafe { value.cast() };
                let d = d.value_of()?;
                dsl::lit(d as i64).cast(DataType::Datetime(TimeUnit::Milliseconds, None))
            } else {
                panic!(
                    "could not convert value {:?} as a Literal",
                    value.coerce_to_string()?.into_utf8()?.into_owned()?
                )
            }
        }
        ValueType::String => String::from_js(value).map(dsl::lit)?,
        ValueType::Null | ValueType::Undefined => dsl::lit(Null {}),
        ValueType::External => {
            let val = unsafe { value.cast::<JsExternal>() };
            if let Ok(series) = cx.env.get_value_external::<Series>(&val) {
                let n = series.name();
                let series = series.clone();
                dsl::lit(series).alias(n)
            } else {
                panic!(
                    "could not convert value {:?} as a Literal",
                    value.coerce_to_string()?.into_utf8()?.into_owned()?
                )
            }
        }
        _ => panic!(
            "could not convert value {:?} as a Literal",
            value.coerce_to_string()?.into_utf8()?.into_owned()?
        ),
    };

    lit.try_into_js(&cx)
}

#[js_function(1)]
pub fn as_str(cx: CallContext) -> JsResult<JsString> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let s = format!("{:#?}", expr);
    cx.env.create_string_from_std(s)
}

#[js_function(1)]
pub fn cast(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let dtype: DataType = params.get_as::<JsDataType>("dtype")?.into();
    let strict = params.get_as::<bool>("strict")?;
    let expr = if strict {
        expr.clone().strict_cast(dtype)
    } else {
        expr.clone().cast(dtype)
    };
    expr.try_into_js(&cx)
}

#[js_function(1)]
pub fn arg_max(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    expr.clone()
        .apply(
            |s| Ok(Series::new(s.name(), &[s.arg_max().map(|idx| idx as u32)])),
            GetOutput::from_type(DataType::UInt32),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn arg_min(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    expr.clone()
        .apply(
            |s| Ok(Series::new(s.name(), &[s.arg_min().map(|idx| idx as u32)])),
            GetOutput::from_type(DataType::UInt32),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn shift_and_fill(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let periods = params.get_as::<i64>("periods")?;
    let fill_value = params.get_external::<Expr>(&cx, "fillValue")?.clone();
    expr.clone()
        .shift_and_fill(periods, fill_value)
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn fill_null_with_strategy(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let strat = params.get_as::<String>("strategy").map(parse_strategy)?;

    expr.clone()
        .apply(move |s| s.fill_null(strat), GetOutput::same_type())
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn take_every(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let n = params.get_as::<usize>("n")?;
    expr.clone()
        .map(move |s: Series| Ok(s.take_every(n)), GetOutput::same_type())
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn slice(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let offset = params.get_as::<i64>("offset")?;
    let length = params.get_as::<usize>("length")?;

    expr.clone().slice(offset, length).try_into_js(&cx)
}

#[js_function(1)]
pub fn over(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let arr: JsObject = params.0.get_named_property("partitionBy")?;
    let len = arr.get_array_length()?;
    let partition_by: Vec<Expr> = (0..len)
        .map(|v| {
            let external = arr
                .get_element_unchecked::<JsExternal>(v)
                .expect("element exists");
            let expr: &mut Expr = cx
                .env
                .get_value_external(&external)
                .expect("value to be an expr");
            expr.clone()
        })
        .collect();
    expr.clone().over(partition_by).try_into_js(&cx)
}

#[js_function(1)]
pub fn str_parse_date(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let fmt = params.get_as::<Option<String>>("fmt")?;
    let function = move |s: Series| {
        let ca = s.utf8()?;
        ca.as_date(fmt.as_deref()).map(|ca| ca.into_series())
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::Date))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_parse_datetime(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let fmt = params.get_as::<Option<String>>("fmt")?;
    let function = move |s: Series| {
        let ca = s.utf8()?;
        ca.as_datetime(fmt.as_deref(), TimeUnit::Milliseconds)
            .map(|ca| ca.into_series())
    };

    expr.clone()
        .map(
            function,
            GetOutput::from_type(DataType::Datetime(TimeUnit::Milliseconds, None)),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_to_uppercase(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let function = move |s: Series| {
        let ca = s.utf8()?;
        Ok(ca.to_uppercase().into_series())
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::UInt32))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_to_lowercase(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let function = move |s: Series| {
        let ca = s.utf8()?;
        Ok(ca.to_lowercase().into_series())
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::UInt32))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_slice(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let start = params.get_as::<i64>("start")?;
    let length = params.get_as::<Option<u64>>("length")?;
    let function = move |s: Series| {
        let ca = s.utf8()?;
        Ok(ca.str_slice(start, length)?.into_series())
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::Utf8))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_lengths(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let function = move |s: Series| {
        let ca = s.utf8()?;
        Ok(ca.str_lengths().into_series())
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::UInt32))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_replace(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let pat = params.get_as::<String>("pat")?;
    let val = params.get_as::<String>("val")?;

    let function = move |s: Series| {
        let ca = s.utf8()?;
        match ca.replace(&pat, &val) {
            Ok(ca) => Ok(ca.into_series()),
            Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
        }
    };

    expr.clone()
        .map(function, GetOutput::same_type())
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_replace_all(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let pat = params.get_as::<String>("pat")?;
    let val = params.get_as::<String>("val")?;

    let function = move |s: Series| {
        let ca = s.utf8()?;
        match ca.replace_all(&pat, &val) {
            Ok(ca) => Ok(ca.into_series()),
            Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
        }
    };

    expr.clone()
        .map(function, GetOutput::same_type())
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_contains(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let pat = params.get_as::<String>("pat")?;

    let function = move |s: Series| {
        let ca = s.utf8()?;
        match ca.contains(&pat) {
            Ok(ca) => Ok(ca.into_series()),
            Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
        }
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::Boolean))
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn str_concat(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let delimiter = params.get_as::<String>("delimiter")?;
    expr.clone().str().concat(&delimiter).try_into_js(&cx)
}

#[js_function(1)]
pub fn str_json_path_match(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let pat = params.get_as::<String>("pat")?;

    let function = move |s: Series| {
        let ca = s.utf8()?;
        match ca.json_path_match(&pat) {
            Ok(ca) => Ok(ca.into_series()),
            Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
        }
    };

    expr.clone()
        .map(function, GetOutput::from_type(DataType::Boolean))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_extract(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let pat = params.get_as::<String>("pat")?;
    let group_index = params.get_as::<usize>("groupIndex")?;

    let function = move |s: Series| {
        let ca = s.utf8()?;
        match ca.extract(&pat, group_index) {
            Ok(ca) => Ok(ca.into_series()),
            Err(e) => Err(PolarsError::ComputeError(format!("{:?}", e).into())),
        }
    };
    expr.clone()
        .map(function, GetOutput::from_type(DataType::Boolean))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn str_split(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let by = params.get_as::<String>("by")?;
    let inclusive: bool = params.get_or("inclusive", false)?;
    if inclusive {
        expr.clone().str().split_inclusive(&by).try_into_js(&cx)
    } else {
        expr.clone().str().split(&by).try_into_js(&cx)
    }
}

#[js_function(1)]
pub fn hex_encode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;

    expr.clone()
        .map(
            move |s| Ok(s.utf8()?.hex_encode().into_series()),
            GetOutput::same_type(),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn hex_decode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let strict = params.get_as::<Option<bool>>("strict")?;
    expr.clone()
        .map(
            move |s| s.utf8()?.hex_decode(strict).map(|s| s.into_series()),
            GetOutput::same_type(),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn base64_encode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    expr.clone()
        .map(
            move |s| Ok(s.utf8()?.base64_encode().into_series()),
            GetOutput::same_type(),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn base64_decode(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let strict = params.get_as::<Option<bool>>("strict")?;
    expr.clone()
        .map(
            move |s| s.utf8()?.base64_decode(strict).map(|s| s.into_series()),
            GetOutput::same_type(),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn strftime(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let fmt = params.get_as::<String>("fmt")?;

    let function = move |s: Series| s.strftime(&fmt);

    expr.clone()
        .map(function, GetOutput::from_type(DataType::Utf8))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn timestamp(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    expr.clone()
        .map(
            |s| s.timestamp().map(|ca| ca.into_series()),
            GetOutput::from_type(DataType::Int64),
        )
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn hash(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let k0 = params.get_as::<u64>("k0")?;
    let k1 = params.get_as::<u64>("k1")?;
    let k2 = params.get_as::<u64>("k2")?;
    let k3 = params.get_as::<u64>("k3")?;
    let function = move |s: Series| {
        let hb = ahash::RandomState::with_seeds(k0, k1, k2, k3);
        Ok(s.hash(hb).into_series())
    };
    expr.clone()
        .map(function, GetOutput::from_type(DataType::UInt64))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn reinterpret(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let signed = params.get_as::<bool>("signed")?;
    let function = move |s: Series| utils::reinterpret(&s, signed);
    let dt = if signed {
        DataType::Int64
    } else {
        DataType::UInt64
    };
    expr.clone()
        .map(function, GetOutput::from_type(dt))
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn exclude(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let columns = params.get_as::<Vec<String>>("columns")?;
    expr.clone().exclude(&columns).try_into_js(&cx)
}

#[js_function(1)]
pub fn reshape(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let dims = params.get_as::<Vec<i64>>("dims")?;
    expr.clone().reshape(&dims).try_into_js(&cx)
}

#[js_function(1)]
pub fn sort_with(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let descending: bool = params.get_or("reverse", false)?;
    let nulls_last: bool = params.get_or("nullsLast", false)?;

    expr.clone()
        .sort_with(SortOptions {
            descending,
            nulls_last,
        })
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn sort_by(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let by = params.get_external_vec::<Expr>(&cx, "by")?;
    let reverse: Vec<bool> = params.get_as("reverse")?;

    expr.clone().sort_by(by, reverse).try_into_js(&cx)
}

#[js_function(1)]
pub fn quantile(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let quantile = params.get_as::<f64>("quantile")?;
    expr.clone()
        .quantile(quantile, QuantileInterpolOptions::default())
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn extend_constant(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    let val = params.get::<JsUnknown>("value")?;
    let n = params.get_as::<usize>("n")?;
    match val.get_type()? {
        ValueType::Undefined | ValueType::Null => expr.apply(
            move |s| s.extend_constant(AnyValue::Null, n),
            GetOutput::same_type(),
        ),
        ValueType::Boolean => {
            let val = bool::from_js(val)?;
            expr.apply(
                move |s| s.extend_constant(AnyValue::Boolean(val), n),
                GetOutput::same_type(),
            )
        }
        ValueType::Number => {
            let f_val = f64::from_js(val)?;
            if f_val.round() == f_val {
                let val = f_val as i64;
                if val > 0 && val < i32::MAX as i64 || val < 0 && val > i32::MIN as i64 {
                    expr.apply(
                        move |s| s.extend_constant(AnyValue::Int32(val as i32), n),
                        GetOutput::same_type(),
                    )
                } else {
                    expr.apply(
                        move |s| s.extend_constant(AnyValue::Int64(val), n),
                        GetOutput::same_type(),
                    )
                }
            } else {
                expr.apply(
                    move |s| s.extend_constant(AnyValue::Float64(f_val), n),
                    GetOutput::same_type(),
                )
            }
        }
        ValueType::String => {
            let val = String::from_js(val)?;
            expr.apply(
                move |s| s.extend_constant(AnyValue::Utf8(&val), n),
                GetOutput::same_type(),
            )
        }
        ValueType::Bigint => {
            let val = u64::from_js(val)?;
            expr.apply(
                move |s| s.extend_constant(AnyValue::UInt64(val), n),
                GetOutput::same_type(),
            )
        }
        ValueType::Object => {
            if val.is_date()? {
                let d: JsDate = unsafe { val.cast() };
                let d = d.value_of()?;
                let d = d as i64;
                expr.apply(
                    move |s| {
                        s.extend_constant(AnyValue::Datetime(d, TimeUnit::Milliseconds, &None), n)
                    },
                    GetOutput::same_type(),
                )
            } else {
                return Err(JsPolarsEr::Other("Unsupported Data type".to_owned()).into());
            }
        }

        _ => return Err(JsPolarsEr::Other("Unsupported Data type".to_owned()).into()),
    }
    .try_into_js(&cx)
}
//
macro_rules! impl_expr {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let expr = params.get_external::<Expr>(&cx, "_expr")?;
            let other = params.get_external::<Expr>(&cx, "other")?.clone();
            expr.clone().$name(other).try_into_js(&cx)
        }
    };
    ($name:ident, $type:ty, $key:expr) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let expr = params.get_external::<Expr>(&cx, "_expr")?;
            let arg = params.get_as::<$type>($key)?;
            expr.clone().$name(arg).try_into_js(&cx)
        }
    };
}

impl_expr!(eq);
impl_expr!(neq);
impl_expr!(gt);
impl_expr!(gt_eq);
impl_expr!(lt_eq);
impl_expr!(lt);
impl_expr!(take);
impl_expr!(fill_null);
impl_expr!(fill_nan);
impl_expr!(filter);
impl_expr!(and);
impl_expr!(xor);
impl_expr!(or);
impl_expr!(is_in);
impl_expr!(repeat_by);
impl_expr!(dot);
impl_expr!(alias, &str, "name");
impl_expr!(sort, bool, "reverse");
impl_expr!(arg_sort, bool, "reverse");
impl_expr!(shift, i64, "periods");
impl_expr!(tail, Option<usize>, "length");
impl_expr!(head, Option<usize>, "length");
impl_expr!(round, u32, "decimals");
impl_expr!(pow, f64, "exponent");
impl_expr!(cumsum, bool, "reverse");
impl_expr!(cummax, bool, "reverse");
impl_expr!(cummin, bool, "reverse");
impl_expr!(cumprod, bool, "reverse");
impl_expr!(cumcount, bool, "reverse");
impl_expr!(prefix, &str, "prefix");
impl_expr!(suffix, &str, "suffix");
impl_expr!(skew, bool, "bias");

macro_rules! impl_no_arg_expr {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let expr = params.get_external::<Expr>(&cx, "_expr")?;
            expr.clone().$name().try_into_js(&cx)
        }
    };
}
impl_no_arg_expr!(not);
impl_no_arg_expr!(is_null);
impl_no_arg_expr!(is_not_null);
impl_no_arg_expr!(is_infinite);
impl_no_arg_expr!(is_finite);
impl_no_arg_expr!(is_nan);
impl_no_arg_expr!(is_not_nan);
impl_no_arg_expr!(min);
impl_no_arg_expr!(max);
impl_no_arg_expr!(mean);
impl_no_arg_expr!(median);
impl_no_arg_expr!(sum);
impl_no_arg_expr!(n_unique);
impl_no_arg_expr!(arg_unique);
impl_no_arg_expr!(unique);
impl_no_arg_expr!(unique_stable);
impl_no_arg_expr!(first);
impl_no_arg_expr!(last);
impl_no_arg_expr!(list);
impl_no_arg_expr!(count);
impl_no_arg_expr!(agg_groups);

impl_no_arg_expr!(backward_fill);
impl_no_arg_expr!(forward_fill);
impl_no_arg_expr!(reverse);
impl_no_arg_expr!(std);
impl_no_arg_expr!(var);
impl_no_arg_expr!(is_unique);
impl_no_arg_expr!(is_first);
impl_no_arg_expr!(explode);
impl_no_arg_expr!(floor);
impl_no_arg_expr!(ceil);
impl_no_arg_expr!(abs);
impl_no_arg_expr!(is_duplicated);

impl_no_arg_expr!(mode);
impl_no_arg_expr!(keep_name);
impl_no_arg_expr!(interpolate);
impl_no_arg_expr!(lower_bound);
impl_no_arg_expr!(upper_bound);

macro_rules! impl_dt_expr {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let expr = params.get_external::<Expr>(&cx, "_expr")?;
            expr.clone().dt().$name().try_into_js(&cx)
        }
    };
}
impl_dt_expr!(year);
impl_dt_expr!(month);
impl_dt_expr!(week);
impl_dt_expr!(weekday);
impl_dt_expr!(day);
impl_dt_expr!(ordinal_day);
impl_dt_expr!(hour);
impl_dt_expr!(minute);
impl_dt_expr!(second);
impl_dt_expr!(nanosecond);

macro_rules! impl_rolling_method {
    ($name:ident) => {
        #[js_function(1)]
        pub fn $name(cx: CallContext) -> JsResult<JsExternal> {
            let params = get_params(&cx)?;
            let expr = params.get_external::<Expr>(&cx, "_expr")?;
            let window_size = params.get_as::<usize>("window_size")?;
            let weights = params.get_as::<Option<Vec<f64>>>("weights")?;
            let min_periods = params.get_as::<usize>("min_periods")?;
            let center = params.get_as::<bool>("center")?;
            let options = RollingOptions {
                window_size,
                weights,
                min_periods,
                center,
            };
            expr.clone().$name(options).try_into_js(&cx)
        }
    };
}
impl_rolling_method!(rolling_max);
impl_rolling_method!(rolling_min);
impl_rolling_method!(rolling_mean);
impl_rolling_method!(rolling_std);
impl_rolling_method!(rolling_var);
impl_rolling_method!(rolling_sum);
impl_rolling_method!(rolling_median);

#[js_function(1)]
pub fn rolling_quantile(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let quantile = params.get_as::<f64>("quantile")?;
    let interpolation: QuantileInterpolOptions =
        params.get_or("interpolation", QuantileInterpolOptions::Nearest)?;
    let window_size: usize = params.get_or("window_size", 2)?;
    let min_periods: usize = params.get_or("min_periods", 2)?;
    let center: bool = params.get_or("center", false)?;
    let weights = params.get_as::<Option<Vec<f64>>>("weights")?;
    let options = RollingOptions {
        window_size,
        weights,
        min_periods,
        center,
    };
    expr.clone()
        .rolling_quantile(quantile, interpolation, options)
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn rolling_skew(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let window_size = params.get_as::<usize>("windowSize")?;
    let bias = params.get_as::<bool>("bias")?;

    expr.clone()
        .rolling_apply_float(window_size, move |ca| {
            ca.clone().into_series().skew(bias).unwrap()
        })
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_lengths(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .lengths()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_max(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .max()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_min(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .min()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_sum(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .sum()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_mean(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .mean()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_sort(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let reverse = params.get_as::<bool>("reverse")?;

    expr.clone().arr().sort(reverse).try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_reverse(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .reverse()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_unique(cx: CallContext) -> JsResult<JsExternal> {
    get_params(&cx)?
        .get_external::<Expr>(&cx, "_expr")?
        .clone()
        .arr()
        .unique()
        .try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_get(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let index = params.get_as::<i64>("index")?;

    expr.clone().arr().get(index).try_into_js(&cx)
}

#[js_function(1)]
pub fn lst_join(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let sep = params.get_as::<String>("separator")?;

    expr.clone().arr().join(&sep).try_into_js(&cx)
}

#[js_function(1)]
pub fn rank(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let method = params.get_as::<String>("method").map(str_to_rankmethod)??;

    expr.clone()
        .rank(RankOptions {
            method,
            descending: false,
        })
        .try_into_js(&cx)
}
#[js_function(1)]
pub fn clip(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let min = params.get_as::<f64>("min")?;
    let max = params.get_as::<f64>("max")?;

    expr.clone().clip(min, max).try_into_js(&cx)
}
#[js_function(1)]
pub fn diff(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let n = params.get_as::<usize>("n")?;
    let null_behavior = params
        .get_as::<String>("nullBehavior")
        .map(str_to_null_behavior)??;

    expr.clone().diff(n, null_behavior).try_into_js(&cx)
}

#[js_function(1)]
pub fn kurtosis(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let expr = params.get_external::<Expr>(&cx, "_expr")?;
    let fisher = params.get_or::<bool>("fisher", true)?;
    let bias = params.get_or::<bool>("bias", false)?;

    expr.clone().kurtosis(fisher, bias).try_into_js(&cx)
}

// When
#[js_function(1)]
pub fn when(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let predicate = params.get_external::<Expr>(&cx, "_expr")?.clone();
    cx.env.create_external(dsl::when(predicate), None)
}

// When::then
#[js_function(1)]
pub fn when_then(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let w = params.get_external::<dsl::When>(&cx, "_when")?.clone();
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    cx.env.create_external(w.then(expr), None)
}

// WhenThen::when
#[js_function(1)]
pub fn when_then_when(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let whenthen = params.get_external::<WhenThen>(&cx, "_when")?.clone();
    let predicate = params.get_external::<Expr>(&cx, "_expr")?.clone();

    cx.env.create_external(whenthen.when(predicate), None)
}

// WhenThen::otherwise
#[js_function(1)]
pub fn when_then_otherwise(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let whenthen = params.get_external::<WhenThen>(&cx, "_when")?.clone();
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();

    cx.env.create_external(whenthen.otherwise(expr), None)
}

// WhenThenThen::when
#[js_function(1)]
pub fn when_then_then_when(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let whenthenthen = params.get_external::<WhenThenThen>(&cx, "_when")?.clone();
    let predicate = params.get_external::<Expr>(&cx, "_expr")?.clone();
    cx.env.create_external(whenthenthen.when(predicate), None)
}

// WhenThenThen::then
#[js_function(1)]
pub fn when_then_then_then(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let whenthenthen = params.get_external::<WhenThenThen>(&cx, "_when")?.clone();
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();
    cx.env.create_external(whenthenthen.then(expr), None)
}

// WhenThenThen::otherwise
#[js_function(1)]
pub fn when_then_then_otherwise(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let whenthenthen = params.get_external::<WhenThenThen>(&cx, "_when")?.clone();
    let expr = params.get_external::<Expr>(&cx, "_expr")?.clone();

    cx.env.create_external(whenthenthen.otherwise(expr), None)
}
