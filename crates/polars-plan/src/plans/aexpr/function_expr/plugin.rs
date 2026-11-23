#![allow(unsafe_op_in_unsafe_fn)]
use std::ffi::CStr;
use std::path::PathBuf;
use std::sync::{LazyLock, RwLock};

use libloading::Library;
use polars_arrow::ffi::{ArrowSchema, import_field_from_c};
use polars_ffi::version_0::{ColumnExport, SeriesExport};
use polars_utils::io::PathIoError;
#[cfg(feature = "python")]
use pyo3::{Python, types::PyAnyMethods};

use super::*;

struct LoadedPlugin {
    library: Library,
    // FFI protocol version.
    ffi_major: u16,
    ffi_minor: u16,
    /// Expression functions by symbol name. Libraries are never unloaded, so these stay valid.
    expr_fns: RwLock<PlHashMap<String, ExprFn>>,
}

#[derive(Clone, Copy)]
enum ExprFn {
    Column(PluginFn<ColumnExport>),
    Series(PluginFn<SeriesExport>),
}

impl LoadedPlugin {
    fn expr_fn(&self, symbol: &str) -> PolarsResult<ExprFn> {
        if let Some(f) = self.expr_fns.read().unwrap().get(symbol) {
            return Ok(*f);
        }

        let f = match self.ffi_major {
            0 => {
                // Plugins taking `Column`s export an additional function which preserves scalar
                // columns.
                let column_fn = unsafe {
                    self.library.get::<PluginFn<ColumnExport>>(
                        format!("_polars_column_plugin_{symbol}").as_bytes(),
                    )
                };
                match column_fn {
                    Ok(f) => ExprFn::Column(*f),
                    Err(_) => ExprFn::Series(unsafe {
                        *self
                            .library
                            .get::<PluginFn<SeriesExport>>(
                                format!("_polars_plugin_{symbol}").as_bytes(),
                            )
                            .unwrap()
                    }),
                }
            },
            major => {
                polars_bail!(ComputeError: "this polars engine doesn't support plugin version: {}", major)
            },
        };
        self.expr_fns.write().unwrap().insert(symbol.to_string(), f);
        Ok(f)
    }
}

static LOADED: LazyLock<RwLock<PlIndexMap<String, Arc<LoadedPlugin>>>> =
    LazyLock::new(Default::default);

fn get_lib(lib: &str) -> PolarsResult<Arc<LoadedPlugin>> {
    if let Some(library) = LOADED.read().unwrap().get(lib) {
        return Ok(library.clone());
    }

    #[cfg(feature = "python")]
    let load_path = if !std::path::Path::new(lib).is_absolute() {
        // Get python virtual environment path
        let prefix = Python::attach(|py| {
            let sys = py.import("sys").unwrap();
            let prefix = sys.getattr("prefix").unwrap();
            prefix.to_string()
        });
        let full_path = std::path::Path::new(&prefix).join(lib);
        full_path.to_string_lossy().into_owned()
    } else {
        lib.to_string()
    };
    #[cfg(not(feature = "python"))]
    let load_path = lib.to_string();

    let library = unsafe {
        Library::new(&load_path).map_err(|e| {
            // Format through `PathIoError` so long paths are truncated consistently with
            // other IO errors, and the full path is only shown with `POLARS_VERBOSE=1`.
            let err = PathIoError {
                path: PathBuf::from(&load_path),
                source: std::io::Error::other(e),
            };
            PolarsError::ComputeError(format!("error loading dynamic library: {err}").into())
        })?
    };
    let version_function: libloading::Symbol<unsafe extern "C" fn() -> u32> = unsafe {
        library
            .get("_polars_plugin_get_version".as_bytes())
            .unwrap()
    };

    let version = unsafe { version_function() };
    let major = (version >> 16) as u16;
    let minor = version as u16;

    let ret = Arc::new(LoadedPlugin {
        library,
        ffi_major: major,
        ffi_minor: minor,
        expr_fns: Default::default(),
    });
    let mut lib_map = LOADED.write().unwrap();
    lib_map.insert(lib.to_string(), ret.clone());
    Ok(ret)
}

fn retrieve_error_msg(lib: &Library) -> String {
    unsafe {
        // SAFETY: _polars_plugin_get_last_error_message returns data stored
        // in a null-terminated thread-local so we immediately clone it.
        let symbol: libloading::Symbol<unsafe extern "C" fn() -> *mut std::os::raw::c_char> =
            lib.get(b"_polars_plugin_get_last_error_message\0").unwrap();
        let msg_ptr = symbol();
        CStr::from_ptr(msg_ptr).to_string_lossy().into_owned()
    }
}

#[doc(hidden)]
pub unsafe fn call_plugin(
    s: &[Column],
    lib: &str,
    symbol: &str,
    kwargs: &[u8],
) -> PolarsResult<Column> {
    use polars_ffi::version_0::*;

    let plugin = get_lib(lib)?;
    let out = match plugin.expr_fn(symbol)? {
        ExprFn::Column(f) => {
            let inputs = s.iter().map(export_column).collect();
            let out = call_plugin_fn(f, inputs, kwargs, ColumnExport::empty());
            (!out.is_null()).then(|| import_column(out))
        },
        ExprFn::Series(f) => {
            let inputs = s
                .iter()
                .map(|c| export_series(c.as_materialized_series()))
                .collect();
            let out = call_plugin_fn(f, inputs, kwargs, SeriesExport::empty());
            (!out.is_null()).then(|| import_series(out).map(Column::from))
        },
    };

    match out {
        Some(out) => out,
        None => {
            let msg = retrieve_error_msg(&plugin.library);
            check_panic(msg.as_ref())?;
            polars_bail!(ComputeError: "the plugin failed with message: {}", msg)
        },
    }
}

// *const E: pointer to the exported inputs
// usize: number of inputs
// *const u8: pointer to &[u8] (kwargs)
// usize: length of the u8 slice
// *mut E: pointer where the return value should be written
// *const CallerContext
type PluginFn<E> = unsafe extern "C" fn(
    *const E,
    usize,
    *const u8,
    usize,
    *mut E,
    *const polars_ffi::version_0::CallerContext,
);

unsafe fn call_plugin_fn<E>(
    f: PluginFn<E>,
    inputs: Vec<E>,
    kwargs: &[u8],
    mut return_value: E,
) -> E {
    let context = polars_ffi::version_0::CallerContext::default();
    f(
        inputs.as_ptr(),
        inputs.len(),
        kwargs.as_ptr(),
        kwargs.len(),
        &mut return_value,
        &context,
    );

    // The inputs get dropped when the ffi side calls the drop callback.
    for e in inputs {
        std::mem::forget(e);
    }
    return_value
}

pub(super) unsafe fn plugin_field(
    fields: &[Field],
    lib: &str,
    symbol: &str,
    kwargs: &[u8],
) -> PolarsResult<Field> {
    let plugin = get_lib(lib)?;

    // we deallocate the fields buffer
    let ffi_fields = fields
        .iter()
        .map(|field| polars_arrow::ffi::export_field_to_c(&field.to_arrow(CompatLevel::newest())))
        .collect::<Vec<_>>()
        .into_boxed_slice();
    let n_args = ffi_fields.len();
    let slice_ptr = ffi_fields.as_ptr();

    let mut return_value = ArrowSchema::empty();
    let return_value_ptr = &mut return_value as *mut ArrowSchema;

    if plugin.ffi_major == 0 {
        match plugin.ffi_minor {
            // The bump from 0 -> 1 was a historic mistake, it should've been a major version.
            0 => {
                let views = fields.iter().any(|field| field.dtype.contains_views());
                polars_ensure!(!views, ComputeError: "cannot call plugin\n\nThis Polars' version has a different 'binary/string' layout. Please compile with latest 'pyo3-polars'");

                // *const ArrowSchema: pointer to heap Box<ArrowSchema>
                // usize: length of the boxed slice
                // *mut ArrowSchema: pointer where the return value can be written
                let symbol: libloading::Symbol<
                    unsafe extern "C" fn(*const ArrowSchema, usize, *mut ArrowSchema),
                > = plugin
                    .library
                    .get((format!("_polars_plugin_field_{symbol}")).as_bytes())
                    .unwrap();
                symbol(slice_ptr, n_args, return_value_ptr);
            },
            // Minor versions above 1 only add capabilities, so they share this signature.
            1.. => {
                // *const ArrowSchema: pointer to heap Box<ArrowSchema>
                // usize: length of the boxed slice
                // *mut ArrowSchema: pointer where the return value can be written
                // *const u8: pointer to &[u8] (kwargs)
                // usize: length of the u8 slice
                let symbol: libloading::Symbol<
                    unsafe extern "C" fn(
                        *const ArrowSchema,
                        usize,
                        *mut ArrowSchema,
                        *const u8,
                        usize,
                    ),
                > = plugin
                    .library
                    .get((format!("_polars_plugin_field_{symbol}")).as_bytes())
                    .unwrap();

                let kwargs_ptr = kwargs.as_ptr();
                let kwargs_len = kwargs.len();

                symbol(slice_ptr, n_args, return_value_ptr, kwargs_ptr, kwargs_len);
            },
        }
        if !return_value.is_null() {
            let arrow_field = import_field_from_c(&return_value)?;
            let out = Field::from(&arrow_field);
            Ok(out)
        } else {
            let msg = retrieve_error_msg(&plugin.library);
            check_panic(msg.as_ref())?;
            polars_bail!(ComputeError: "the plugin failed with message: {}", msg)
        }
    } else {
        polars_bail!(ComputeError: "this Polars engine doesn't support plugin version: {}", plugin.ffi_major)
    }
}

fn check_panic(msg: &str) -> PolarsResult<()> {
    polars_ensure!(msg != "PANIC", ComputeError: "the plugin panicked\n\nThe message is suppressed. Set POLARS_VERBOSE=1 to send the panic message to stderr.");
    Ok(())
}
