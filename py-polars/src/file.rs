// Credits to https://github.com/omerbenamram/pyo3-file
use std::borrow::Borrow;
use std::fs::File;
use std::io;
use std::io::{BufReader, Cursor, Read, Seek, SeekFrom, Write};

use polars::io::mmap::MmapBytesReader;
use pyo3::exceptions::{PyFileNotFoundError, PyTypeError};
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::prelude::resolve_homedir;

#[derive(Clone)]
pub struct PyFileLikeObject {
    inner: PyObject,
}

/// Wraps a `PyObject`, and implements read, seek, and write for it.
impl PyFileLikeObject {
    /// Creates an instance of a `PyFileLikeObject` from a `PyObject`.
    /// To assert the object has the required methods methods,
    /// instantiate it with `PyFileLikeObject::require`
    pub fn new(object: PyObject) -> Self {
        PyFileLikeObject { inner: object }
    }

    pub fn as_buffer(&self) -> std::io::Cursor<Vec<u8>> {
        let data = self.as_file_buffer().into_inner();
        std::io::Cursor::new(data)
    }

    pub fn as_file_buffer(&self) -> Cursor<Vec<u8>> {
        let buf = Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method(py, "read", (), None)
                .expect("no read method found");

            let bytes: &PyBytes = bytes
                .downcast(py)
                .expect("Expecting to be able to downcast into bytes from read result.");

            bytes.as_bytes().to_vec()
        });

        Cursor::new(buf)
    }

    /// Take a Python buffer and extends it lifetime to static.
    ///
    /// # Safety
    /// It also returns the bytes PyObject. As long as that object is held, the lifetime is valid
    /// as the destructor is not called.
    pub unsafe fn as_file_buffer_ref(&self) -> (Cursor<&'static [u8]>, PyObject) {
        Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method(py, "read", (), None)
                .expect("no read method found");

            let ref_bytes: &PyBytes = bytes
                .downcast(py)
                .expect("Expecting to be able to downcast into bytes from read result.");
            let buf = ref_bytes.as_bytes();

            let static_buf = std::mem::transmute::<&[u8], &'static [u8]>(buf);
            (Cursor::new(static_buf), bytes)
        })
    }

    /// Same as `PyFileLikeObject::new`, but validates that the underlying
    /// python object has a `read`, `write`, and `seek` methods in respect to parameters.
    /// Will return a `TypeError` if object does not have `read`, `seek`, and `write` methods.
    pub fn with_requirements(
        object: PyObject,
        read: bool,
        write: bool,
        seek: bool,
    ) -> PyResult<Self> {
        Python::with_gil(|py| {
            if read && object.getattr(py, "read").is_err() {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Object does not have a .read() method.",
                ));
            }

            if seek && object.getattr(py, "seek").is_err() {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Object does not have a .seek() method.",
                ));
            }

            if write && object.getattr(py, "write").is_err() {
                return Err(PyErr::new::<PyTypeError, _>(
                    "Object does not have a .write() method.",
                ));
            }

            Ok(PyFileLikeObject::new(object))
        })
    }
}

/// Extracts a string repr from, and returns an IO error to send back to rust.
fn pyerr_to_io_err(e: PyErr) -> io::Error {
    Python::with_gil(|py| {
        let e_as_object: PyObject = e.into_py(py);

        match e_as_object.call_method(py, "__str__", (), None) {
            Ok(repr) => match repr.extract::<String>(py) {
                Ok(s) => io::Error::new(io::ErrorKind::Other, s),
                Err(_e) => io::Error::new(io::ErrorKind::Other, "An unknown error has occurred"),
            },
            Err(_) => io::Error::new(io::ErrorKind::Other, "Err doesn't have __str__"),
        }
    })
}

impl Read for PyFileLikeObject {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let bytes = self
                .inner
                .call_method(py, "read", (buf.len(),), None)
                .map_err(pyerr_to_io_err)?;

            let bytes: &PyBytes = bytes
                .downcast(py)
                .expect("Expecting to be able to downcast into bytes from read result.");

            buf.write_all(bytes.as_bytes())?;

            bytes.len().map_err(pyerr_to_io_err)
        })
    }
}

impl Write for PyFileLikeObject {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let pybytes = PyBytes::new(py, buf);

            let number_bytes_written = self
                .inner
                .call_method(py, "write", (pybytes,), None)
                .map_err(pyerr_to_io_err)?;

            number_bytes_written.extract(py).map_err(pyerr_to_io_err)
        })
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        Python::with_gil(|py| {
            self.inner
                .call_method(py, "flush", (), None)
                .map_err(pyerr_to_io_err)?;

            Ok(())
        })
    }
}

impl Seek for PyFileLikeObject {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, io::Error> {
        Python::with_gil(|py| {
            let (whence, offset) = match pos {
                SeekFrom::Start(i) => (0, i as i64),
                SeekFrom::Current(i) => (1, i),
                SeekFrom::End(i) => (2, i),
            };

            let new_position = self
                .inner
                .call_method(py, "seek", (offset, whence), None)
                .map_err(pyerr_to_io_err)?;

            new_position.extract(py).map_err(pyerr_to_io_err)
        })
    }
}

pub trait FileLike: Read + Write + Seek + Sync + Send {}

impl FileLike for File {}
impl FileLike for PyFileLikeObject {}
impl MmapBytesReader for PyFileLikeObject {}

pub enum EitherRustPythonFile {
    Py(PyFileLikeObject),
    Rust(BufReader<File>),
}

///
/// # Arguments
/// * `truncate` - open or create a new file.
pub fn get_either_file(py_f: PyObject, truncate: bool) -> PyResult<EitherRustPythonFile> {
    Python::with_gil(|py| {
        if let Ok(pstring) = py_f.downcast::<PyString>(py) {
            let rstring = pstring.to_string();
            let str_slice: &str = rstring.borrow();
            let f = if truncate {
                BufReader::new(File::create(str_slice)?)
            } else {
                match File::open(str_slice) {
                    Ok(file) => BufReader::new(file),
                    Err(_e) => {
                        return Err(PyErr::new::<PyFileNotFoundError, _>(format!(
                            "No such file or directory: {str_slice}",
                        )))
                    }
                }
            };
            Ok(EitherRustPythonFile::Rust(f))
        } else {
            let f = PyFileLikeObject::with_requirements(py_f, true, true, true)?;
            Ok(EitherRustPythonFile::Py(f))
        }
    })
}

pub fn get_file_like(f: PyObject, truncate: bool) -> PyResult<Box<dyn FileLike>> {
    use EitherRustPythonFile::*;
    match get_either_file(f, truncate)? {
        Py(f) => Ok(Box::new(f)),
        Rust(f) => Ok(Box::new(f.into_inner())),
    }
}

pub fn get_mmap_bytes_reader<'a>(py_f: &'a PyAny) -> PyResult<Box<dyn MmapBytesReader + 'a>> {
    // bytes object
    if let Ok(bytes) = py_f.downcast::<PyBytes>() {
        Ok(Box::new(Cursor::new(bytes.as_bytes())))
    }
    // string so read file
    else if let Ok(pstring) = py_f.downcast::<PyString>() {
        let s = pstring.to_string();
        let p = std::path::Path::new(&s);
        let p = resolve_homedir(p);
        let f = match File::open(p) {
            Ok(file) => file,
            Err(_e) => {
                return Err(PyErr::new::<PyFileNotFoundError, _>(format!(
                    "No such file or directory: {s}",
                )))
            }
        };
        Ok(Box::new(f))
    }
    // a normal python file: with open(...) as f:.
    else if py_f.getattr("read").is_ok() {
        // we can still get a file name, inform the user of possibly wrong API usage.
        if py_f.getattr("name").is_ok() {
            eprint!("Polars found a filename. \
            Ensure you pass a path to the file instead of a python file object when possible for best \
            performance.")
        }
        // a bytesIO
        if let Ok(bytes) = py_f.call_method0("getvalue") {
            let bytes = bytes.downcast::<PyBytes>()?;
            Ok(Box::new(Cursor::new(bytes.as_bytes())))
        }
        // don't really know what we got here, just read.
        else {
            let f = Python::with_gil(|py| {
                PyFileLikeObject::with_requirements(py_f.to_object(py), true, false, true)
            })?;
            Ok(Box::new(f))
        }
    }
    // don't really know what we got here, just read.
    else {
        let f = Python::with_gil(|py| {
            PyFileLikeObject::with_requirements(py_f.to_object(py), true, false, true)
        })?;
        Ok(Box::new(f))
    }
}
