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
        let bytes = self.env.create_buffer_copy(buf.to_owned()).unwrap();
        let js_buff = bytes.into_raw();
        stream_write.call(Some(&self.inner), &[js_buff]).unwrap();
        Ok(buf.len())
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        // JS write streams do not have a 'flush'
        Ok(())
    }
}

// impl Read for JsFileLike<'_> {
//   fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
//     let on: JsFunction = self.inner.get_named_property("on").expect("no error");
//     let data_str = self
//       .env
//       .create_string("data")
//       .expect("no error")
//       .into_unknown();
//     let callback = self
//       .env
//       .create_function_from_closure("callback", |cx| {
//         println!("inside callback");
//         cx.env.get_undefined()
//       })
//       .unwrap()
//       .into_unknown();
//     let args = vec![data_str, callback];
//     on.call(Some(&self.inner), &args).unwrap();
//     todo!()
//   }
// }
