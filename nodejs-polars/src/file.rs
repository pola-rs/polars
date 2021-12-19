use napi::{JsFunction, JsObject};
use std::io;
use std::io::Write;

pub struct JsFileLike<'a> {
    pub inner: JsObject,
    pub env: &'a napi::Env,
}

impl<'a> JsFileLike<'a> {
    pub fn new(obj: JsObject, env: &'a napi::Env) -> Self {
        JsFileLike {
            inner: obj,
            env: env,
        }
    }
}

impl Write for JsFileLike<'_> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        let stream_write: JsFunction = self.inner.get_named_property("write").unwrap();
        let bytes = self.env.create_buffer_with_data(buf.to_owned()).unwrap();
        let js_buff = bytes.into_raw();
        stream_write.call(Some(&self.inner), &[js_buff]).unwrap();
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        // JS write streams do not have a 'flush'
        Ok(())
    }
}
