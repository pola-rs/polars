use crate::conversion::prelude::*;
use crate::prelude::JsResult;
use crate::series::JsSeries;
use napi::*;
use polars::lazy::dsl;
// use polars::lazy::dsl::Operator;
use polars::prelude::*;

#[derive(Clone)]
pub struct JsExpr {
    pub inner: dsl::Expr,
}
impl From<dsl::Expr> for JsExpr {
    fn from(expr: dsl::Expr) -> Self {
        JsExpr { inner: expr }
    }
}






#[derive(Clone)]
pub struct When {
    predicate: JsExpr,
}

#[derive(Clone)]
pub struct WhenThen {
    predicate: JsExpr,
    then: JsExpr,
}

#[derive(Clone)]
pub struct WhenThenThen {
    inner: dsl::WhenThenThen,
}



#[js_function(1)]
pub fn col(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let n = params.get_as::<&str>("name")?;
    let expr: JsExpr = dsl::col(n).into();
    expr.try_into_js(&cx)
}

#[js_function(1)]
pub fn cols(cx: CallContext) -> JsResult<JsExternal> {
    let params = get_params(&cx)?;
    let names = params.get_as::<Vec<String>>("name")?;
    let expr: JsExpr = dsl::cols(names).into();
    expr.try_into_js(&cx)
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
        ValueType::String => String::from_js(value).map(dsl::lit)?,
        ValueType::Null | ValueType::Undefined => dsl::lit(Null {}),
        ValueType::External => {
            let val = unsafe { value.cast::<JsExternal>() };
            if let Ok(series) = cx.env.get_value_external::<JsSeries>(&val) {
                dsl::lit((&series).series.clone())
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

    JsExpr::from(lit).try_into_js(&cx)
}