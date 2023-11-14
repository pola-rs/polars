use super::*;

/// An FFI exported `Series`.
#[repr(C)]
pub struct SeriesExport {
    field: *mut ArrowSchema,
    // A double ptr, so we can easily release the buffer
    // without dropping the arrays.
    arrays: *mut *mut ArrowArray,
    len: usize,
    release: Option<unsafe extern "C" fn(arg1: *mut SeriesExport)>,
    private_data: *mut std::os::raw::c_void,
}

impl SeriesExport {
    pub fn empty() -> Self {
        Self {
            field: std::ptr::null_mut(),
            arrays: std::ptr::null_mut(),
            len: 0,
            release: None,
            private_data: std::ptr::null_mut(),
        }
    }

    pub fn is_null(&self) -> bool {
        self.private_data.is_null()
    }
}

impl Drop for SeriesExport {
    fn drop(&mut self) {
        if let Some(release) = self.release {
            unsafe { release(self) }
        }
    }
}

// callback used to drop [SeriesExport] when it is exported.
unsafe extern "C" fn c_release_series_export(e: *mut SeriesExport) {
    if e.is_null() {
        return;
    }
    let e = &mut *e;
    let private = Box::from_raw(e.private_data as *mut PrivateData);
    for ptr in private.arrays.iter() {
        // drop the box, not the array
        let _ = Box::from_raw(*ptr as *mut ManuallyDrop<ArrowArray>);
    }

    e.release = None;
}

pub fn export_series(s: &Series) -> SeriesExport {
    let field = ArrowField::new(s.name(), s.dtype().to_arrow(), true);
    let schema = Box::new(ffi::export_field_to_c(&field));
    let mut arrays = s
        .chunks()
        .iter()
        .map(|arr| Box::into_raw(Box::new(ffi::export_array_to_c(arr.clone()))))
        .collect::<Box<_>>();
    let len = arrays.len();
    let ptr = arrays.as_mut_ptr();
    SeriesExport {
        field: schema.as_ref() as *const ArrowSchema as *mut ArrowSchema,
        arrays: ptr,
        len,
        release: Some(c_release_series_export),
        private_data: Box::into_raw(Box::new(PrivateData { arrays, schema }))
            as *mut std::os::raw::c_void,
    }
}

/// # Safety
/// `SeriesExport` must be valid
pub unsafe fn import_series(e: SeriesExport) -> PolarsResult<Series> {
    let field = ffi::import_field_from_c(&(*e.field))?;

    let pointers = std::slice::from_raw_parts_mut(e.arrays, e.len);
    let chunks = pointers
        .iter()
        .map(|ptr| {
            let arr = std::ptr::read(*ptr);
            import_array(arr, &(*e.field))
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    Series::try_from((field.name.as_str(), chunks))
}

/// # Safety
/// `SeriesExport` must be valid
pub unsafe fn import_series_buffer(e: *mut SeriesExport, len: usize) -> PolarsResult<Vec<Series>> {
    let mut out = Vec::with_capacity(len);
    for i in 0..len {
        let e = std::ptr::read(e.add(i));
        out.push(import_series(e)?)
    }
    Ok(out)
}

/// Passed to an expression.
/// This contains information for the implementer of the expression on what it is allowed to do.
#[derive(Copy, Clone, Debug, Default)]
#[repr(C)]
pub struct CallerContext {
    /// The expression may implement their own parallelism.
    pub parallelized: bool,
}

#[cfg(test)]
mod test {
    use polars_core::prelude::*;

    use super::*;

    #[test]
    fn test_ffi() {
        let s = Series::new("a", [1, 2]);
        let e = export_series(&s);

        unsafe {
            assert_eq!(import_series(e).unwrap(), s);
        };
    }
}
