use crate::conversion::wrap::*;
use crate::error::JsPolarsEr;
use crate::prelude::JsResult;
use napi::{CallContext, JsObject, Result};
use polars::frame::NullStrategy;
use polars::prelude::*;
use polars_core::prelude::FillNullStrategy;
use polars_core::series::ops::NullBehavior;

pub fn get_params(cx: &CallContext) -> Result<WrappedObject> {
    Ok(cx.get::<JsObject>(0)?.into())
}

pub(crate) fn parse_strategy(strat: String) -> FillNullStrategy {
    match strat.as_str() {
        "backward" => FillNullStrategy::Backward,
        "forward" => FillNullStrategy::Forward,
        "min" => FillNullStrategy::Min,
        "max" => FillNullStrategy::Max,
        "mean" => FillNullStrategy::Mean,
        "zero" => FillNullStrategy::Zero,
        "one" => FillNullStrategy::One,
        s => panic!("Strategy {} not supported", s),
    }
}
pub(crate) fn str_to_rankmethod(method: String) -> JsResult<RankMethod> {
    let method = match method.as_str() {
        "min" => RankMethod::Min,
        "max" => RankMethod::Max,
        "average" => RankMethod::Average,
        "dense" => RankMethod::Dense,
        "ordinal" => RankMethod::Ordinal,
        "random" => RankMethod::Random,
        _ => {
            return Err(
                JsPolarsEr::Other("use one of 'avg, min, max, dense, ordinal'".to_string()).into(),
            )
        }
    };
    Ok(method)
}
pub(crate) fn str_to_null_strategy(strategy: &str) -> JsResult<NullStrategy> {
    let strategy = match strategy {
        "ignore" => NullStrategy::Ignore,
        "propagate" => NullStrategy::Propagate,
        _ => {
            return Err(napi::Error::from_reason(
                "use one of 'ignore', 'propagate'".to_string(),
            ))
        }
    };
    Ok(strategy)
}
pub(crate) fn str_to_null_behavior(null_behavior: String) -> JsResult<NullBehavior> {
    let null_behavior = match null_behavior.as_str() {
        "drop" => NullBehavior::Drop,
        "ignore" => NullBehavior::Ignore,
        _ => return Err(JsPolarsEr::Other("use one of 'drop', 'ignore'".to_string()).into()),
    };
    Ok(null_behavior)
}

pub fn reinterpret(s: &Series, signed: bool) -> polars::prelude::Result<Series> {
    match (s.dtype(), signed) {
        (DataType::UInt64, true) => {
            let ca = s.u64().unwrap();
            Ok(ca.reinterpret_signed().into_series())
        }
        (DataType::UInt64, false) => Ok(s.clone()),
        (DataType::Int64, false) => {
            let ca = s.i64().unwrap();
            Ok(ca.reinterpret_unsigned().into_series())
        }
        (DataType::Int64, true) => Ok(s.clone()),
        _ => Err(PolarsError::ComputeError(
            "reinterpret is only allowed for 64bit integers dtype, use cast otherwise".into(),
        )),
    }
}
