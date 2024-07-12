use std::borrow::Cow;
use std::fs::{self, File};
use std::io;
use std::io::{Cursor, ErrorKind, Read, Seek, SeekFrom, Write};
#[cfg(target_family = "unix")]
use std::os::fd::{FromRawFd, RawFd};
use std::path::PathBuf;

use polars::io::mmap::MmapBytesReader;
use polars_error::{polars_err, polars_warn};
use pyo3::exceptions::PyTypeError;
use pyo3::prelude::*;
use pyo3::types::{PyBytes, PyString};

use crate::error::PyPolarsErr;
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
                .call_method_bound(py, "read", (), None)
                .expect("no read method found");

            let bytes: &Bound<'_, PyBytes> = bytes
                .downcast_bound(py)
                .expect("Expecting to be able to downcast into bytes from read result.");

            bytes.as_bytes().to_vec()
        });

        Cursor::new(buf)
    }

    /// Validates that the underlying
    /// python object has a `read`, `write`, and `seek` methods in respect to parameters.
    /// Will return a `TypeError` if object does not have `read`, `seek`, and `write` methods.
    pub fn ensure_requirements(
        object: &Bound<PyAny>,
        read: bool,
        write: bool,
        seek: bool,
    ) -> PyResult<()> {
        if read && object.getattr("read").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .read() method.",
            ));
        }

        if seek && object.getattr("seek").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .seek() method.",
            ));
        }

        if write && object.getattr("write").is_err() {
            return Err(PyErr::new::<PyTypeError, _>(
                "Object does not have a .write() method.",
            ));
        }

        Ok(())
    }
}

/// Extracts a string repr from, and returns an IO error to send back to rust.
fn pyerr_to_io_err(e: PyErr) -> io::Error {
    Python::with_gil(|py| {
        let e_as_object: PyObject = e.into_py(py);

        match e_as_object.call_method_bound(py, "__str__", (), None) {
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
                .call_method_bound(py, "read", (buf.len(),), None)
                .map_err(pyerr_to_io_err)?;

            let opt_bytes = bytes.downcast_bound::<PyBytes>(py);

            if let Ok(bytes) = opt_bytes {
                buf.write_all(bytes.as_bytes())?;

                bytes.len().map_err(pyerr_to_io_err)
            } else if let Ok(s) = bytes.downcast_bound::<PyString>(py) {
                let s = s.to_cow().map_err(pyerr_to_io_err)?;
                buf.write_all(s.as_bytes())?;
                Ok(s.len())
            } else {
                Err(io::Error::new(
                    ErrorKind::InvalidInput,
                    polars_err!(InvalidOperation: "could not read from input"),
                ))
            }
        })
    }
}

impl Write for PyFileLikeObject {
    fn write(&mut self, buf: &[u8]) -> Result<usize, io::Error> {
        Python::with_gil(|py| {
            let pybytes = PyBytes::new_bound(py, buf);

            let number_bytes_written = self
                .inner
                .call_method_bound(py, "write", (pybytes,), None)
                .map_err(pyerr_to_io_err)?;

            number_bytes_written.extract(py).map_err(pyerr_to_io_err)
        })
    }

    fn flush(&mut self) -> Result<(), io::Error> {
        Python::with_gil(|py| {
            self.inner
                .call_method_bound(py, "flush", (), None)
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
                .call_method_bound(py, "seek", (offset, whence), None)
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
    Rust(File),
}

impl EitherRustPythonFile {
    pub fn into_dyn(self) -> Box<dyn FileLike> {
        match self {
            EitherRustPythonFile::Py(f) => Box::new(f),
            EitherRustPythonFile::Rust(f) => Box::new(f),
        }
    }
}

fn get_either_file_and_path(
    py_f: PyObject,
    write: bool,
) -> PyResult<(EitherRustPythonFile, Option<PathBuf>)> {
    Python::with_gil(|py| {
        let py_f = py_f.into_bound(py);
        if let Ok(s) = py_f.extract::<Cow<str>>() {
            let file_path = std::path::Path::new(&*s);
            let file_path = resolve_homedir(file_path);
            let f = if write {
                File::create(&file_path)?
            } else {
                polars_utils::open_file(&file_path).map_err(PyPolarsErr::from)?
            };
            Ok((EitherRustPythonFile::Rust(f), Some(file_path)))
        } else {
            let io = py.import_bound("io").unwrap();
            let is_utf8_encoding = |py_f: &Bound<PyAny>| -> PyResult<bool> {
                let encoding = py_f.getattr("encoding")?;
                let encoding = encoding.extract::<Cow<str>>()?;
                Ok(encoding.eq_ignore_ascii_case("utf-8") || encoding.eq_ignore_ascii_case("utf8"))
            };
            #[cfg(target_family = "unix")]
            if let Some(fd) = (py_f.is_exact_instance(&io.getattr("FileIO").unwrap())
                || (py_f.is_exact_instance(&io.getattr("BufferedReader").unwrap())
                    || py_f.is_exact_instance(&io.getattr("BufferedWriter").unwrap())
                    || py_f.is_exact_instance(&io.getattr("BufferedRandom").unwrap())
                    || py_f.is_exact_instance(&io.getattr("BufferedRWPair").unwrap())
                    || (py_f.is_exact_instance(&io.getattr("TextIOWrapper").unwrap())
                        && is_utf8_encoding(&py_f)?))
                    && if write {
                        // invalidate read buffer
                        py_f.call_method0("flush").is_ok()
                    } else {
                        // flush write buffer
                        py_f.call_method1("seek", (0, 1)).is_ok()
                    })
            .then(|| {
                py_f.getattr("fileno")
                    .and_then(|fileno| fileno.call0())
                    .and_then(|fileno| fileno.extract::<libc::c_int>())
                    .ok()
            })
            .flatten()
            .map(|fileno| unsafe {
                // `File::from_raw_fd()` takes the ownership of the file descriptor.
                // When the File is dropped, it closes the file descriptor.
                // This is undesired - the Python file object will become invalid.
                // Therefore, we duplicate the file descriptor here.
                // Closing the duplicated file descriptor will not close
                // the original file descriptor;
                // and the status, e.g. stream position, is still shared with
                // the original file descriptor.
                // We use `F_DUPFD_CLOEXEC` here instead of `dup()`
                // because it also sets the `O_CLOEXEC` flag on the duplicated file descriptor,
                // which `dup()` clears.
                // `open()` in both Rust and Python automatically set `O_CLOEXEC` flag;
                // it prevents leaking file descriptors across processes,
                // and we want to be consistent with them.
                // `F_DUPFD_CLOEXEC` is defined in POSIX.1-2008
                // and is present on all alive UNIX(-like) systems.
                libc::fcntl(fileno, libc::F_DUPFD_CLOEXEC, 0)
            })
            .filter(|fileno| *fileno != -1)
            .map(|fileno| fileno as RawFd)
            {
                return Ok((
                    EitherRustPythonFile::Rust(unsafe { File::from_raw_fd(fd) }),
                    // This works on Linux and BSD with procfs mounted,
                    // otherwise it fails silently.
                    fs::canonicalize(format!("/proc/self/fd/{fd}")).ok(),
                ));
            }

            // BytesIO is relatively fast, and some code relies on it.
            if !py_f.is_exact_instance(&io.getattr("BytesIO").unwrap()) {
                polars_warn!("Polars found a filename. \
                Ensure you pass a path to the file instead of a python file object when possible for best \
                performance.");
            }
            // Unwrap TextIOWrapper
            // Allow subclasses to allow things like pytest.capture.CaptureIO
            let py_f = if py_f
                .is_instance(&io.getattr("TextIOWrapper").unwrap())
                .unwrap_or_default()
            {
                if !is_utf8_encoding(&py_f)? {
                    return Err(PyPolarsErr::from(
                        polars_err!(InvalidOperation: "file encoding is not UTF-8"),
                    )
                    .into());
                }
                // XXX: we have to clear buffer here.
                // Is there a better solution?
                if write {
                    py_f.call_method0("flush")?;
                } else {
                    py_f.call_method1("seek", (0, 1))?;
                }
                py_f.getattr("buffer")?
            } else {
                py_f
            };
            PyFileLikeObject::ensure_requirements(&py_f, !write, write, !write)?;
            let f = PyFileLikeObject::new(py_f.to_object(py));
            Ok((EitherRustPythonFile::Py(f), None))
        }
    })
}

///
/// # Arguments
/// * `write` - open for writing; will truncate existing file and create new file if not.
pub fn get_either_file(py_f: PyObject, write: bool) -> PyResult<EitherRustPythonFile> {
    Ok(get_either_file_and_path(py_f, write)?.0)
}

pub fn get_file_like(f: PyObject, truncate: bool) -> PyResult<Box<dyn FileLike>> {
    Ok(get_either_file(f, truncate)?.into_dyn())
}

/// If the give file-like is a BytesIO, read its contents.
pub fn read_if_bytesio(py_f: Bound<PyAny>) -> Bound<PyAny> {
    if py_f.getattr("read").is_ok() {
        let Ok(bytes) = py_f.call_method0("getvalue") else {
            return py_f;
        };
        if bytes.downcast::<PyBytes>().is_ok() {
            return bytes.clone();
        }
    }
    py_f
}

/// Create reader from PyBytes or a file-like object. To get BytesIO to have
/// better performance, use read_if_bytesio() before calling this.
pub fn get_mmap_bytes_reader<'a>(
    py_f: &'a Bound<'a, PyAny>,
) -> PyResult<Box<dyn MmapBytesReader + 'a>> {
    get_mmap_bytes_reader_and_path(py_f).map(|t| t.0)
}

pub fn get_mmap_bytes_reader_and_path<'a>(
    py_f: &'a Bound<'a, PyAny>,
) -> PyResult<(Box<dyn MmapBytesReader + 'a>, Option<PathBuf>)> {
    // bytes object
    if let Ok(bytes) = py_f.downcast::<PyBytes>() {
        Ok((Box::new(Cursor::new(bytes.as_bytes())), None))
    }
    // string so read file
    else {
        match get_either_file_and_path(py_f.to_object(py_f.py()), false)? {
            (EitherRustPythonFile::Rust(f), path) => Ok((Box::new(f), path)),
            (EitherRustPythonFile::Py(f), path) => Ok((Box::new(f), path)),
        }
    }
}
