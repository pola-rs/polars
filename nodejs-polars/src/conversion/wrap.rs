use crate::conversion::prelude::*;
use crate::error::JsPolarsEr;
use crate::series::JsSeries;
use napi::{CallContext, JsExternal, JsObject, JsUnknown, Result};
use polars::chunked_array::object::PolarsObjectSafe;

#[derive(Debug)]
pub struct Wrap<T>(pub T);

impl<T> Clone for Wrap<T>
where
    T: Clone,
{
    fn clone(&self) -> Self {
        Wrap(self.0.clone())
    }
}
impl<T> From<T> for Wrap<T> {
    fn from(t: T) -> Self {
        Wrap(t)
    }
}

pub struct WrappedObject(pub(crate) JsObject);

impl From<JsObject> for WrappedObject {
    fn from(h: JsObject) -> Self {
        Self(h)
    }
}

impl WrappedObject {
    pub fn get_as<V: FromJsUnknown>(&self, key: &str) -> Result<V> {
        let v: JsUnknown = self.0.get_named_property(key)?;
        V::from_js(v)
    }

    pub fn get_or<V: FromJsUnknown>(&self, key: &str, default: V) -> Result<V> {
        let v: JsUnknown = self.0.get_named_property(key)?;
        Option::<V>::from_js(v).map(|opt| opt.unwrap_or(default))
    }
    pub fn get<V: napi::NapiValue>(&self, key: &str) -> Result<V> {
        self.0.get_named_property::<V>(key)
    }
    pub fn get_arr(&self, key: &str) -> Result<(Vec<WrappedValue>, usize)> {
        let jsv: JsObject = self.0.get_named_property(key)?;
        let len = jsv.get_array_length()?;
        let mut v: Vec<WrappedValue> = Vec::with_capacity(len as usize);
        for idx in 0..len {
            let item: WrappedValue = jsv.get_element_unchecked::<JsUnknown>(idx)?.into();
            v.push(item)
        }
        Ok((v, len as usize))
    }

    pub fn get_series<'a>(&'a self, cx: &'a CallContext, key: &str) -> Result<&'a JsSeries> {
        let v: JsExternal = self.0.get_named_property(key)?;
        let s: &JsSeries = cx.env.get_value_external(&v)?;
        Ok(s)
    }

    pub fn get_external<'a, Out: 'static>(
        &'a self,
        cx: &'a CallContext,
        key: &str,
    ) -> Result<&Out> {
        let v: JsExternal = self.0.get_named_property(key)?;
        let s: &Out = cx.env.get_value_external(&v)?;
        Ok(s)
    }

    pub fn get_external_mut<'a, Out: 'static>(
        &'a self,
        cx: &'a CallContext,
        key: &str,
    ) -> Result<&mut Out> {
        let v: JsExternal = self.0.get_named_property(key)?;
        let s: &mut Out = cx.env.get_value_external(&v)?;
        Ok(s)
    }
    pub fn get_external_vec<'a, Out: 'static + Clone>(
        &'a self,
        cx: &'a CallContext,
        key: &str,
    ) -> Result<Vec<Out>> {
        let jsv: JsObject = self.0.get_named_property(key)?;
        let len = jsv.get_array_length()?;
        (0..len)
            .map(|idx| {
                let ext = jsv.get_element_unchecked::<JsExternal>(idx)?;
                let s: &mut Out = cx.env.get_value_external(&ext)?;
                Ok(s.clone())
            })
            .collect()
    }
}

pub struct WrappedValue(pub(crate) JsUnknown);

impl From<JsUnknown> for WrappedValue {
    fn from(jsv: JsUnknown) -> Self {
        Self(jsv)
    }
}

impl WrappedValue {
    pub fn extract<V: FromJsUnknown>(self) -> Result<V> {
        V::from_js(self.0)
    }

    pub fn to_array(self) -> Result<Vec<WrappedValue>> {
        if self.0.is_array()? {
            let obj: JsObject = unsafe { self.0.cast() };
            let len = obj.get_array_length()?;
            let mut v: Vec<WrappedValue> = Vec::with_capacity(len as usize);
            for idx in 0..len {
                let item: WrappedValue = obj.get_element_unchecked::<JsUnknown>(idx)?.into();
                v.push(item)
            }
            Ok(v)
        } else {
            Err(JsPolarsEr::Other("Must be array type".to_owned()).into())
        }
    }
}

pub struct ObjectValue {
    pub inner: JsUnknown,
}

impl From<&dyn PolarsObjectSafe> for &ObjectValue {
    fn from(val: &dyn PolarsObjectSafe) -> Self {
        unsafe { &*(val as *const dyn PolarsObjectSafe as *const ObjectValue) }
    }
}
