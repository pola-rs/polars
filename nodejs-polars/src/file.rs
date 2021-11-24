// use napi::{CallContext, JsBuffer, JsFunction, JsObject, JsString, JsUnknown, ValueType};
// use std::fs::File;
// use std::io;
// use std::io::prelude::*;
// use std::io::{Cursor, Read, Seek, SeekFrom, Write};

// use crate::prelude::JsResult;

// pub struct JsFileLike {
//   inner: JsObject,
//   env: napi::Env,
// }

// impl JsFileLike {
//   pub fn new(obj: JsObject, env: napi::Env) -> Self {
//     JsFileLike {
//       inner: obj,
//       env: env,
//     }
//   }
// }

// //   // pub fn as_buffer(self) -> std::io::Cursor<Vec<u8>> {
// //   //   let data = self.as_file_buffer().into_inner();
// //   //   Cursor::new(data)
// //   // }

// //   // pub fn as_file_buffer(self) -> Cursor<Vec<u8>> {
// //   //   let bytes = self.inner.into_ref().expect("unexpected error");
// //   //   let bytes: &[u8] = bytes.as_ref();

// //   //   let buf = bytes.to_vec();

// //   //   Cursor::new(buf)
// //   // }
// // }


// impl Read for JsFileLike {
//   fn read(&mut self, buf: &mut [u8]) -> Result<usize, io::Error> {
//     let on: JsFunction = self.inner.get_named_property("on").expect("no error");
//     let data_str = self
//       .env
//       .create_string("data")
//       .expect("no error")
//       .into_unknown();
    
//     let console = self
//       .env
//       .create_function_from_closure("handle", |cx| {
//         let chunk: JsString = cx.get(0)?;
//         let chunk = chunk.into_utf8()?;
//         let chunk = chunk.as_slice();
//         buf.write_all(chunk)?;
        
//         cx.env.get_undefined()
//       })
//       .expect("no error creating fn")
//       .into_unknown();

//     todo!()
//   }
// }

// // // pub fn get_file_like(f: JsUnknown, truncate: bool) -> JsResult<Box<dyn FileLike>> {
// // //   match f.get_type()? {
// // //     ValueType::String => {
// // //       let s: JsString = unsafe { f.cast() };
// // //       let s = s.into_utf8()?;
// // //       let v: &str = s.as_str()?;
// // //       let f = if truncate {
// // //         File::create(v)?
// // //       } else {
// // //         File::open(v)?
// // //       };
// // //       Ok(Box::new(f))
// // //     }
// // //     ValueType::Object => {
// // //       let buff: JsBuffer = unsafe { f.cast() };
// // //       let r = buff.into_ref()?;
// // //       let buff: &[u8] = r.as_ref();
// // //       let b = Cursor::new(buff);

// // //       // Ok(Box::new(b))
// // //       todo!()
// // //     }
// // //     _ => Err(napi::Error::from_reason(
// // //       "unsupported type, must be path or buffer".to_owned(),
// // //     )),
// // //   }

// // //   // todo!()
// // // }

// // // pub trait FileLike: Read + Write + Seek + Sync + Send {}

// // // impl FileLike for File {}
