// use crate::conversion::str_to_null_behavior;
// use crate::conversion::str_to_rankmethod;
// use crate::conversion::Wrap;
// use crate::utils::str_to_polarstype;
// use crate::{console_log, log};
use polars::prelude::*;
use wasm_bindgen::JsCast;

// use super::{error::JsPolarsErr, JsResult};
// use crate::conversion::FromJsValue;
use crate::{extern_iterator, extern_struct};

use std::ops::Deref;
use wasm_bindgen::prelude::*;

#[wasm_bindgen(js_name=Series)]
#[repr(transparent)]
pub struct JsSeries {
    pub(crate) series: Series,
}

impl JsSeries {
    pub(crate) fn new(series: Series) -> Self {
        JsSeries { series }
    }
}

impl From<Series> for JsSeries {
    fn from(series: Series) -> Self {
        Self { series }
    }
}

// impl wasm_bindgen::convert::FromWasmAbi for JsSeries {

// }
impl Deref for JsSeries {
    type Target = Series;

    fn deref(&self) -> &Self::Target {
        &self.series
    }
}

#[wasm_bindgen]
extern "C" {
    #[wasm_bindgen(typescript_type = "Series")]
    pub type ExternSeries;

    #[wasm_bindgen(typescript_type = "any")]
    pub type ExternAnyValue;

    #[wasm_bindgen(method, getter = ptr)]
    fn ptr(this: &ExternSeries) -> f64;

    #[wasm_bindgen(static_method_of = ExternSeries)]
    fn wrap(ptr: u32) -> ExternSeries;

    #[wasm_bindgen(typescript_type = "Series[]")]
    pub type SeriesArray;

}

extern_struct!(ExternSeries, JsSeries);
extern_iterator!(SeriesArray, ExternSeries, JsSeries);

#[wasm_bindgen(js_class=Series)]
impl JsSeries {}

// pub fn reinterpret(s: &Series, signed: bool) -> polars::prelude::Result<Series> {
//     match (s.dtype(), signed) {
//         (DataType::UInt64, true) => {
//             let ca = s.u64().unwrap();
//             Ok(ca.reinterpret_signed().into_series())
//         }
//         (DataType::UInt64, false) => Ok(s.clone()),
//         (DataType::Int64, false) => {
//             let ca = s.i64().unwrap();
//             Ok(ca.reinterpret_unsigned().into_series())
//         }
//         (DataType::Int64, true) => Ok(s.clone()),
//         _ => Err(PolarsError::ComputeError(
//             "reinterpret is only allowed for 64bit integers dtype, use cast otherwise".into(),
//         )),
//     }
// }
pub(crate) fn to_series_collection(iter: js_sys::Iterator) -> Vec<Series> {
    let cols: Vec<Series> = iter
        .into_iter()
        .map(|jsv| {
            let jsv = jsv.unwrap();
            let key = JsValue::from_str("ptr");
            let ptr = js_sys::Reflect::get(&jsv, &key).unwrap();
            let n: f64 = js_sys::Number::unchecked_from_js(ptr).into();
            let ser: JsSeries = unsafe { JsSeries::from_abi(n as u32) };
            ser.series
        })
        .collect();
    cols
}

// pub(crate) fn to_jsseries_collection(s: Vec<Series>) -> Vec<u32> {
//     use wasm_bindgen::convert::IntoWasmAbi;
//     let s: Vec<u32> = s
//         .into_iter()
//         .map(move |series| {
//             let js_ser = JsSeries { series };

//             js_ser.into_abi()
//         })
//         .collect();

//     s
//     // todo!()
// }
