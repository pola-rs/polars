// Credits to https://github.com/omerbenamram/pyo3-file
use crate::prelude::resolve_homedir;
use polars::io::mmap::MmapBytesReader;
#[cfg(feature = "parquet")]
use polars::io::parquet::SliceableCursor;
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};
use std::borrow::Borrow;
use std::fs::File;
use std::io;
use std::io::{Cursor, Read, Seek, SeekFrom, Write};

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

    #[cfg(feature = "parquet")]
    pub fn as_slicable_buffer(&self) -> SliceableCursor {
        let data = self.as_file_buffer().into_inner();
        SliceableCursor::new(data)
    }

    pub fn as_file_buffer(&self) -> Cursor<Vec<u8>> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let bytes = self
            .inner
            .call_method(py, "read", (), None)
            .expect("no read method found");

        let bytes: &PyBytes = bytes
            .cast_as(py)
            .expect("Expecting to be able to downcast into bytes from read result.");

        let buf = bytes.as_bytes().to_vec();

        Cursor::new(buf)
    }

    /// Take a Python buffer and extends it lifetime to static.
    ///
    /// # Safety
    /// It also returns the bytes PyObject. As long as that object is held, the lifetime is valid
    /// as the destructor is not called.
    pub unsafe fn as_file_buffer_ref(&self) -> (Cursor<&'static [u8]>, PyObject) {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let bytes = self
            .inner
            .call_method(py, "read", (), None)
            .expect("no read method found");

        let ref_bytes: &PyBytes = bytes
            .cast_as(py)
            .expect("Expecting to be able to downcast into bytes from read result.");
        let buf = ref_bytes.as_bytes();

        let static_buf = std::mem::transmute::<&[u8], &'static [u8]>(buf);
        (Cursor::new(static_buf), bytes)
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
        let gil = Python::acquire_gil();
        let py = gil.python();

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
    }
}

/// Extracts a string repr from, and returns an IO error to send back to rust.
fn pyerr_to_io_err(e: PyErr) -> io::Error {
    let gil = Python::acquire_gil();
    let py = gil.python();
    let e_as_object: PyObject = e.into_py(py);

    match e_as_object.call_method(py, "__str__", (), None) {
        Ok(repr) => match repr.extract::<String>(py) {
            Ok(s) => io::Error::new(io::ErrorKind::Other, s),
            Err(_e) => io::Error::new(io::ErrorKind::Other, "An unknown error has occurred"),
        },
        Err(_) => io::Error::new(io::ErrorKind::Other, "Err doesn't have __str__"),
    }
}

impl Read for PyFileLikeObject {
    fn read(&mut self, mut buf: &mut [u8]) -> Result<usize, io::Error> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let bytes = self
            .inner
            .call_method(py, "read", (buf.len(),), None)
            .map_err(pyerr_to_io_err)?;

        let bytes: &PyBytes = bytes
            .cast_as(py)
            .expect("Expecting to be able to downcast into bytes from read result.");

        buf.write_all(bytes.as_bytes())?;

        bytes.len().map_err(pyerr_to_io_err)
    }
}

impl Write for PyFileLikeObject {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let pybytes = PyBytes::new(py, buf);

        let number_bytes_written = self
            .inner
            .call_method(py, "write", (pybytes,), None)
            .map_err(pyerr_to_io_err)?;

        number_bytes_written.extract(py).map_err(pyerr_to_io_err)
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        self.inner
            .call_method(py, "flush", (), None)
            .map_err(pyerr_to_io_err)?;

        Ok(())
    }
}

impl Seek for PyFileLikeObject {
    fn seek(&mut self, pos: SeekFrom) -> Result<u64, io::Error> {
        let gil = Python::acquire_gil();
        let py = gil.python();

        let (whence, offset) = match pos {
            SeekFrom::Start(i) => (0, i as i64),
            SeekFrom::Current(i) => (1, i as i64),
            SeekFrom::End(i) => (2, i as i64),
        };

        let new_position = self
            .inner
            .call_method(py, "seek", (offset, whence), None)
            .map_err(pyerr_to_io_err)?;

        new_position.extract(py).map_err(pyerr_to_io_err)
    }
}

pub trait FileLike: Read + Write + Seek + Sync + Send {}

impl FileLike for File {}
impl FileLike for PyFileLikeObject {}
impl MmapBytesReader for PyFileLikeObject {}

pub enum EitherRustPythonFile {
    Py(PyFileLikeObject),
    Rust(File),
}

///
/// # Arguments
/// * `truncate` - open or create a new file.
pub fn get_either_file(py_f: PyObject, truncate: bool) -> PyResult<EitherRustPythonFile> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    if let Ok(pstring) = py_f.cast_as::<PyString>(py) {
        let rstring = pstring.to_string();
        let str_slice: &str = rstring.borrow();
        let f = if truncate {
            File::create(str_slice)?
        } else {
            File::open(str_slice)?
        };
        Ok(EitherRustPythonFile::Rust(f))
    } else {
        let f = PyFileLikeObject::with_requirements(py_f, true, true, true)?;
        Ok(EitherRustPythonFile::Py(f))
    }
}

pub fn get_file_like(f: PyObject, truncate: bool) -> PyResult<Box<dyn FileLike>> {
    use EitherRustPythonFile::*;
    match get_either_file(f, truncate)? {
        Py(f) => Ok(Box::new(f)),
        Rust(f) => Ok(Box::new(f)),
    }
}

pub fn get_mmap_bytes_reader<'a>(py_f: &'a PyAny) -> PyResult<Box<dyn MmapBytesReader + 'a>> {
    let gil = Python::acquire_gil();
    let py = gil.python();

    // string so read file
    if let Ok(pstring) = py_f.downcast::<PyString>() {
        let s = pstring.to_string();
        let p = std::path::Path::new(&s);
        let p = resolve_homedir(p);
        let f = File::open(&p)?;
        Ok(Box::new(f))
    }
    // bytes object
    else if let Ok(bytes) = py_f.downcast::<PyBytes>() {
        Ok(Box::new(Cursor::new(bytes.as_bytes())))
    }
    // a normal python file: with open(...) as f:.
    else if py_f.getattr("read").is_ok() {
        // we van still get a file name so open the file instead of go through read
        if let Ok(filename) = py_f.getattr("name") {
            let filename = filename.downcast::<PyString>()?;
            let f = File::open(filename.to_str()?)?;
            Ok(Box::new(f))
        }
        // don't really know what we got here, just read.
        else {
            let f = PyFileLikeObject::with_requirements(py_f.to_object(py), true, false, true)?;
            Ok(Box::new(f))
        }
    // a bytesIO
    } else if let Ok(bytes) = py_f.call_method0("getvalue") {
        let bytes = bytes.downcast::<PyBytes>()?;
        Ok(Box::new(Cursor::new(bytes.as_bytes())))
    }
    // don't really know what we got here, just read.
    else {
        let f = PyFileLikeObject::with_requirements(py_f.to_object(py), true, false, true)?;
        Ok(Box::new(f))
    }
}
