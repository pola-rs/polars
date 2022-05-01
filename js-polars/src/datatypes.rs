use polars::prelude::DataType;

#[repr(u32)]
pub enum JsDataType {
    Int8,
    Int16,
    Int32,
    Int64,
    UInt8,
    UInt16,
    UInt32,
    UInt64,
    Float32,
    Float64,
    Bool,
    Utf8,
    List,
    Date,
    Datetime,
    Time,
    // Object,
    Categorical,
}

impl JsDataType {
    pub fn to_string(self) -> String {
        match self {
            JsDataType::Int8 => "Int8",
            JsDataType::Int16 => "Int16",
            JsDataType::Int32 => "Int32",
            JsDataType::Int64 => "Int64",
            JsDataType::UInt8 => "UInt8",
            JsDataType::UInt16 => "UInt16",
            JsDataType::UInt32 => "UInt32",
            JsDataType::UInt64 => "UInt64",
            JsDataType::Float32 => "Float32",
            JsDataType::Float64 => "Float64",
            JsDataType::Bool => "Bool",
            JsDataType::Utf8 => "Utf8",
            JsDataType::List => "List",
            JsDataType::Date => "Date",
            JsDataType::Datetime => "Datetime",
            JsDataType::Time => "Time",
            // JsDataType::Object => "Object",
            JsDataType::Categorical => "Categorical",
        }
        .to_owned()
    }
}

impl From<&DataType> for JsDataType {
    fn from(dt: &DataType) -> Self {
        use JsDataType::*;
        match dt {
            DataType::Int8 => Int8,
            DataType::Int16 => Int16,
            DataType::Int32 => Int32,
            DataType::Int64 => Int64,
            DataType::UInt8 => UInt8,
            DataType::UInt16 => UInt16,
            DataType::UInt32 => UInt32,
            DataType::UInt64 => UInt64,
            DataType::Float32 => Float32,
            DataType::Float64 => Float64,
            DataType::Boolean => Bool,
            DataType::Utf8 => Utf8,
            DataType::List(_) => List,
            DataType::Date => Date,
            DataType::Datetime(_, _) => Datetime,
            DataType::Time => Time,

            // DataType::Object(_) => Object,
            DataType::Categorical(_) => Categorical,
            _ => {
                panic!("null or unknown not expected here")
            }
        }
    }
}

pub enum TypedArrayType {
    Int8(js_sys::Int8Array),
    Uint8(js_sys::Uint8Array),
    Uint8Clamped(js_sys::Uint8ClampedArray),
    Int16(js_sys::Int16Array),
    Uint16(js_sys::Uint16Array),
    Int32(js_sys::Int32Array),
    Uint32(js_sys::Uint32Array),
    Float32(js_sys::Float32Array),
    Float64(js_sys::Float64Array),
    BigInt64(js_sys::BigInt64Array),
    BigUint64(js_sys::BigUint64Array),
}

impl From<wasm_bindgen::JsValue> for TypedArrayType {
    fn from(v: wasm_bindgen::JsValue) -> Self {
        use wasm_bindgen::JsCast;

        if js_sys::Int8Array::instanceof(&v) {
            TypedArrayType::Int8(v.into())
        } else if js_sys::Uint8Array::instanceof(&v) {
            TypedArrayType::Uint8(v.into())
        } else if js_sys::Uint8ClampedArray::instanceof(&v) {
            TypedArrayType::Uint8Clamped(v.into())
        } else if js_sys::Int16Array::instanceof(&v) {
            TypedArrayType::Int16(v.into())
        } else if js_sys::Uint16Array::instanceof(&v) {
            TypedArrayType::Uint16(v.into())
        } else if js_sys::Int32Array::instanceof(&v) {
            TypedArrayType::Int32(v.into())
        } else if js_sys::Uint32Array::instanceof(&v) {
            TypedArrayType::Uint32(v.into())
        } else if js_sys::Float32Array::instanceof(&v) {
            TypedArrayType::Float32(v.into())
        } else if js_sys::Float64Array::instanceof(&v) {
            TypedArrayType::Float64(v.into())
        } else if js_sys::BigInt64Array::instanceof(&v) {
            TypedArrayType::BigInt64(v.into())
        } else if js_sys::BigUint64Array::instanceof(&v) {
            TypedArrayType::BigUint64(v.into())
        } else {
            panic!("unknown dtype")
        }
    }
}
