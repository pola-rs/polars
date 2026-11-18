use std::ffi::{CStr, CString};
use std::ops::DerefMut;
use std::panic::AssertUnwindSafe;

use polars_error::{PolarsError, PolarsResult, polars_bail, polars_err};

use super::{
    ArrowArray, ArrowArrayStream, ArrowSchema, export_array_to_c, export_field_to_c,
    import_array_from_c, import_field_from_c,
};
use crate::array::Array;
use crate::datatypes::Field;

impl Drop for ArrowArrayStream {
    fn drop(&mut self) {
        match self.release {
            None => (),
            Some(release) => unsafe { release(self) },
        };
    }
}

unsafe impl Send for ArrowArrayStream {}

impl ArrowArrayStream {
    /// Creates an empty [`ArrowArrayStream`] used to import from a producer.
    pub fn empty() -> Self {
        Self {
            get_schema: None,
            get_next: None,
            get_last_error: None,
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }
}

unsafe fn handle_error(iter: &mut ArrowArrayStream) -> PolarsError {
    let error = unsafe { (iter.get_last_error.unwrap())(&mut *iter) };

    if error.is_null() {
        return polars_err!(ComputeError: "got unspecified external error");
    }

    let error = unsafe { CStr::from_ptr(error) };
    polars_err!(ComputeError: "got external error: {}", error.to_string_lossy())
}

/// Implements an iterator of [`Array`] consumed from the [C stream interface](https://arrow.apache.org/docs/format/CStreamInterface.html).
pub struct ArrowArrayStreamReader<Iter: DerefMut<Target = ArrowArrayStream>> {
    iter: Iter,
    field: Field,
}

impl<Iter: DerefMut<Target = ArrowArrayStream>> ArrowArrayStreamReader<Iter> {
    /// Returns a new [`ArrowArrayStreamReader`]
    /// # Error
    /// Errors iff the [`ArrowArrayStream`] is out of specification,
    /// or was already released prior to calling this function.
    ///
    /// # Safety
    /// This method is intrinsically `unsafe` since it assumes that the `ArrowArrayStream`
    /// contains a valid Arrow C stream interface.
    /// In particular:
    /// * The `ArrowArrayStream` fulfills the invariants of the C stream interface
    /// * The schema `get_schema` produces fulfills the C data interface
    pub unsafe fn try_new(mut iter: Iter) -> PolarsResult<Self> {
        if iter.release.is_none() {
            polars_bail!(InvalidOperation: "the C stream was already released")
        };

        if iter.get_next.is_none() {
            polars_bail!(InvalidOperation: "the C stream must contain a non-null get_next")
        };

        if iter.get_last_error.is_none() {
            polars_bail!(InvalidOperation: "The C stream MUST contain a non-null get_last_error")
        };

        let mut field = ArrowSchema::empty();
        let status = if let Some(f) = iter.get_schema {
            unsafe { (f)(&mut *iter, &mut field) }
        } else {
            polars_bail!(InvalidOperation:
                            "The C stream MUST contain a non-null get_schema"
            )
        };

        if status != 0 {
            return Err(unsafe { handle_error(&mut iter) });
        }

        let field = unsafe { import_field_from_c(&field)? };

        Ok(Self { iter, field })
    }

    /// Returns the field provided by the stream
    pub fn field(&self) -> &Field {
        &self.field
    }

    /// Advances this iterator by one array
    /// # Error
    /// Errors iff:
    /// * The C stream interface returns an error
    /// * The C stream interface returns an invalid array (that we can identify, see Safety below)
    ///
    /// # Safety
    /// Calling this iterator's `next` assumes that the [`ArrowArrayStream`] produces arrow arrays
    /// that fulfill the C data interface
    pub unsafe fn next(&mut self) -> Option<PolarsResult<Box<dyn Array>>> {
        let mut array = ArrowArray::empty();
        let status = unsafe { (self.iter.get_next.unwrap())(&mut *self.iter, &mut array) };

        if status != 0 {
            return Some(Err(unsafe { handle_error(&mut self.iter) }));
        }

        // last paragraph of https://arrow.apache.org/docs/format/CStreamInterface.html#c.ArrowArrayStream.get_next
        array.release?;

        // SAFETY: assumed from the C stream interface
        unsafe { import_array_from_c(array, self.field.dtype.clone()) }
            .map(Some)
            .transpose()
    }
}

struct PrivateData {
    iter: Box<dyn Iterator<Item = PolarsResult<Box<dyn Array>>> + Send>,
    field: Field,
    error: Option<CString>,
    /// Whether a callback panicked, which ends the stream as `iter` is left in an unknown state.
    panicked: bool,
}

/// The status of a failed callback, as no status of the interface itself is ever this.
const CALLBACK_ERROR: i32 = 2001;

fn error_cstring(message: &str) -> CString {
    CString::new(message.replace('\0', "\\0")).unwrap()
}

/// Run the body of a stream callback, storing a panic as the stream's error, as unwinding out of
/// an `extern "C"` function aborts the process.
unsafe fn catch_callback(
    iter: *mut ArrowArrayStream,
    body: impl FnOnce(&mut PrivateData) -> i32,
) -> i32 {
    if iter.is_null() {
        return CALLBACK_ERROR;
    }
    let private = (*iter).private_data as *mut PrivateData;
    if (*private).panicked {
        return CALLBACK_ERROR;
    }
    match std::panic::catch_unwind(AssertUnwindSafe(|| body(&mut *private))) {
        Ok(status) => status,
        Err(payload) => {
            let message = payload
                .downcast_ref::<&str>()
                .copied()
                .or_else(|| payload.downcast_ref::<String>().map(String::as_str))
                .unwrap_or("unknown panic");
            (*private).error = Some(error_cstring(&format!(
                "panic in an Arrow C stream callback: {message}"
            )));
            (*private).panicked = true;
            CALLBACK_ERROR
        },
    }
}

unsafe extern "C" fn get_next(iter: *mut ArrowArrayStream, array: *mut ArrowArray) -> i32 {
    catch_callback(iter, |private| match private.iter.next() {
        Some(Ok(item)) => {
            // check that the array has the same dtype as field
            let item_dt = item.dtype();
            let expected_dt = private.field.dtype();
            if item_dt != expected_dt {
                private.error = Some(error_cstring(&format!(
                    "The iterator produced an item of data type {item_dt:?} but the producer expects data type {expected_dt:?}"
                )));
                return CALLBACK_ERROR;
            }

            std::ptr::write(array, export_array_to_c(item));

            private.error = None;
            0
        },
        Some(Err(err)) => {
            private.error = Some(error_cstring(&err.to_string()));
            CALLBACK_ERROR
        },
        None => {
            let a = ArrowArray::empty();
            std::ptr::write_unaligned(array, a);
            private.error = None;
            0
        },
    })
}

unsafe extern "C" fn get_schema(iter: *mut ArrowArrayStream, schema: *mut ArrowSchema) -> i32 {
    catch_callback(iter, |private| {
        std::ptr::write(schema, export_field_to_c(&private.field));
        0
    })
}

unsafe extern "C" fn get_last_error(iter: *mut ArrowArrayStream) -> *const ::std::os::raw::c_char {
    if iter.is_null() {
        return std::ptr::null();
    }
    let private = &mut *((*iter).private_data as *mut PrivateData);

    private
        .error
        .as_ref()
        .map(|x| x.as_ptr())
        .unwrap_or(std::ptr::null())
}

unsafe extern "C" fn release(iter: *mut ArrowArrayStream) {
    if iter.is_null() {
        return;
    }
    let _ = Box::from_raw((*iter).private_data as *mut PrivateData);
    (*iter).release = None;
    // private drops automatically
}

/// Exports an iterator to the [C stream interface](https://arrow.apache.org/docs/format/CStreamInterface.html)
///
/// The iterator must be `Send`, as the consumer may read the stream from any thread.
pub fn export_iterator(
    iter: Box<dyn Iterator<Item = PolarsResult<Box<dyn Array>>> + Send>,
    field: Field,
) -> ArrowArrayStream {
    let private_data = Box::new(PrivateData {
        iter,
        field,
        error: None,
        panicked: false,
    });

    ArrowArrayStream {
        get_schema: Some(get_schema),
        get_next: Some(get_next),
        get_last_error: Some(get_last_error),
        release: Some(release),
        private_data: Box::into_raw(private_data) as *mut ::std::os::raw::c_void,
    }
}
