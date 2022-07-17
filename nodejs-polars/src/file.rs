use napi::bindgen_prelude::{Buffer, Null};
use napi::threadsafe_function::*;
use napi::{Either, JsFunction, JsObject};
use std::io;
use std::io::Write;

pub struct JsFileLike<'a> {
    pub inner: JsObject,
    pub env: &'a napi::Env,
}

pub struct JsWriteStream<'a> {
    pub inner: JsObject,
    pub env: &'a napi::Env,
}

impl<'a> JsFileLike<'a> {
    pub fn new(inner: JsObject, env: &'a napi::Env) -> Self {
        JsFileLike { inner, env }
    }
}

impl<'a> JsWriteStream<'a> {
    pub fn new(inner: JsObject, env: &'a napi::Env) -> Self {
        JsWriteStream { inner, env }
    }
}
pub struct ThreadsafeWriteable {
    pub inner: ThreadsafeFunction<Either<Buffer, Null>, ErrorStrategy::CalleeHandled>,
}

impl Write for ThreadsafeWriteable {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        let tsfn = self.inner.clone();
        tsfn.call(
            Ok(Either::A(buf.into())),
            ThreadsafeFunctionCallMode::Blocking,
        );
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        // JS write streams do not have a 'flush'
        Ok(())
    }
}
impl Write for JsFileLike<'_> {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        let stream_write: JsFunction = self.inner.get_named_property("push").unwrap();
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

impl Write for JsWriteStream<'_> {
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
