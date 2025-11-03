use std::mem::MaybeUninit;
use std::ptr::NonNull;

use arrow::ffi::ArrowSchema;
use ::polars_ffi::version_0::SeriesExport;

use super::{DataPtr, StatePtr, StatefulUdfTrait};

#[derive(Clone)]
#[repr(C)]
pub struct VTable {
    pub(super) _initialize:
        unsafe extern "C" fn(DataPtr, NonNull<ArrowSchema>, NonNull<MaybeUninit<StatePtr>>) -> u32,
    pub(super) _new_empty:
        unsafe extern "C" fn(DataPtr, StatePtr, NonNull<MaybeUninit<StatePtr>>) -> u32,
    pub(super) _reset: unsafe extern "C" fn(DataPtr, StatePtr) -> u32,

    pub(super) _insert: unsafe extern "C" fn(
        DataPtr,
        StatePtr,
        *mut SeriesExport,
        usize,
        NonNull<bool>,
        NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32,
    pub(super) _finalize: unsafe extern "C" fn(
        DataPtr,
        StatePtr,
        NonNull<bool>,
        NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32,
    pub(super) _combine: unsafe extern "C" fn(DataPtr, StatePtr, StatePtr) -> u32,

    pub(super) _to_field: unsafe extern "C" fn(
        DataPtr,
        NonNull<ArrowSchema>,
        NonNull<MaybeUninit<ArrowSchema>>,
    ) -> u32,
    pub(super) _flags: unsafe extern "C" fn(DataPtr, NonNull<u64>) -> u32,
    pub(super) _format: unsafe extern "C" fn(DataPtr, NonNull<*const u8>, NonNull<usize>) -> u32,
    pub(super) _drop_state: unsafe extern "C" fn(StatePtr) -> u32,
    pub(super) _drop_data: unsafe extern "C" fn(DataPtr) -> u32,

    pub(super) _get_error: unsafe extern "C" fn(NonNull<*const u8>, NonNull<isize>),
}

#[repr(u32)]
enum ReturnValue {
    Ok = 0,

    Panic = 1,

    InvalidOperation = 2,
    ComputeError = 3,
    ShapeMismatch = 4,
    AssertionError = 5,

    OtherError = 6,
}

impl VTable {
    pub const fn new<Data: StatefulUdfTrait>() -> Self {
        Self {
            _initialize: _callee::initialize::<Data>,
            _new_empty: _callee::new_empty::<Data>,
            _reset: _callee::reset::<Data>,

            _insert: _callee::insert::<Data>,
            _finalize: _callee::finalize::<Data>,
            _combine: _callee::combine::<Data>,

            _to_field: _callee::to_field::<Data>,
            _flags: _callee::flags::<Data>,
            _format: _callee::format::<Data>,
            _drop_state: _callee::drop_state::<Data>,
            _drop_data: _callee::drop_data::<Data>,

            _get_error: _callee::get_error,
        }
    }
}

impl From<u32> for ReturnValue {
    fn from(value: u32) -> Self {
        match value {
            0 => Self::Ok,

            1 => Self::Panic,

            2 => Self::InvalidOperation,
            3 => Self::ComputeError,
            4 => Self::ShapeMismatch,
            5 => Self::AssertionError,

            _ => Self::OtherError,
        }
    }
}

mod _callee {
    use std::cell::RefCell;
    use std::ffi::c_int;
    use std::mem::MaybeUninit;
    use std::panic::{AssertUnwindSafe, UnwindSafe};
    use std::ptr::NonNull;

    use ::polars_core::prelude::{Field, Schema};
    use ::polars_core::series::Series;
    use ::polars_error::PolarsResult;
    use arrow::datatypes::ArrowDataType;
    use arrow::ffi::{ArrowSchema, export_field_to_c, import_field_from_c};
    use polars_core::prelude::CompatLevel;
    use polars_core::schema::SchemaExt;
    use polars_error::{PolarsError, polars_bail};
    use ::polars_ffi::version_0::{SeriesExport, export_series, import_series, import_series_buffer};

    use super::super::StatefulUdfTrait;
    use super::{DataPtr, ReturnValue, StatePtr};

    thread_local! { static ERROR_MESSAGE: RefCell<Option<Box<str>>> = RefCell::new(None); }

    pub unsafe extern "C" fn get_error(ptr: NonNull<*const u8>, length: NonNull<isize>) {
        ERROR_MESSAGE.with_borrow(|v| {
            let (n_ptr, n_len) = v.as_ref().map_or((std::ptr::null(), -1), |v| {
                (v.as_ptr(), v.len().try_into().unwrap_or(0))
            });
            unsafe {
                ptr.write(n_ptr);
            }
            unsafe {
                length.write(n_len);
            }
        })
    }

    unsafe fn import_pl_schema(schema: NonNull<ArrowSchema>) -> PolarsResult<Schema> {
        let fields = unsafe { schema.as_ref() };
        let fields = unsafe { import_field_from_c(fields) }?;
        let ArrowDataType::Struct(fields) = fields.dtype else {
            polars_bail!(ComputeError: "received invalid schema in plugin FFI");
        };
        let fields = arrow::datatypes::ArrowSchema::from_iter(fields);
        Ok(Schema::from_arrow_schema(&fields))
    }

    fn wrap_callee_function(f: impl FnOnce() -> PolarsResult<()> + UnwindSafe) -> u32 {
        let result = std::panic::catch_unwind(|| f());

        let set_msg = match &result {
            Ok(Ok(_)) => Ok(()),
            Err(panic) => std::panic::catch_unwind(|| {
                ERROR_MESSAGE.with_borrow_mut(|b| *b = Some("panicked".into()))
            }),
            Ok(Err(err)) => {
                let (kind, msg) = match err {
                    PolarsError::InvalidOperation(msg)
                    | PolarsError::ComputeError(msg)
                    | PolarsError::ShapeMismatch(msg)
                    | PolarsError::AssertionError(msg) => (None, Some(msg)),
                    PolarsError::Duplicate(msg) => (Some("DuplicateError"), Some(msg)),
                    PolarsError::IO { error, msg } => (Some("IO"), None),
                    PolarsError::NoData(msg) => (Some("NoData"), Some(msg)),
                    PolarsError::OutOfBounds(msg) => (Some("OutOfBounds"), Some(msg)),
                    PolarsError::SchemaFieldNotFound(msg) => {
                        (Some("SchemaFieldNotFound"), Some(msg))
                    },
                    PolarsError::SchemaMismatch(msg) => (Some("SchemaMismatch"), Some(msg)),
                    PolarsError::SQLInterface(msg) => (Some("SQLInterface"), Some(msg)),
                    PolarsError::SQLSyntax(msg) => (Some("SQLSyntax"), Some(msg)),
                    PolarsError::StringCacheMismatch(msg) => {
                        (Some("StringCacheMismatch"), Some(msg))
                    },
                    PolarsError::StructFieldNotFound(msg) => {
                        (Some("StructFieldNotFound"), Some(msg))
                    },
                    PolarsError::ColumnNotFound(msg) => (Some("ColumnNotFound"), Some(msg)),
                    PolarsError::Context { error, msg } => (Some("unknown"), None),
                    #[cfg(feature = "python")]
                    PolarsError::Python { error } => (Some("python"), None),
                };

                let msg = msg.map(|m| m.as_ref());
                match (kind, msg) {
                    (None, None) => Ok(()),
                    (Some(m), None) | (None, Some(m)) => std::panic::catch_unwind(move || {
                        ERROR_MESSAGE.with_borrow_mut(|b| *b = Some(format!("{m}").into()))
                    }),
                    (Some(kind), Some(msg)) => std::panic::catch_unwind(move || {
                        ERROR_MESSAGE
                            .with_borrow_mut(|b| *b = Some(format!("{kind}: {msg}").into()))
                    }),
                }
            },
        };

        let Ok(_) = set_msg else {
            // If we panic while setting error message, just abort.
            ::std::process::abort();
        };

        (match result {
            Ok(Ok(_)) => ReturnValue::Ok,
            Err(_) => ReturnValue::Panic,
            Ok(Err(err)) => match err {
                PolarsError::InvalidOperation(_) => ReturnValue::InvalidOperation,
                PolarsError::ComputeError(_) => ReturnValue::ComputeError,
                PolarsError::ShapeMismatch(_) => ReturnValue::ShapeMismatch,
                PolarsError::AssertionError(_) => ReturnValue::AssertionError,
                _ => ReturnValue::OtherError,
            },
        }) as u32
    }

    pub unsafe extern "C" fn initialize<Data: StatefulUdfTrait>(
        data: DataPtr,
        fields: NonNull<ArrowSchema>,
        state: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let fields = unsafe { import_pl_schema(fields) }?;
            let out = data.initialize(&fields)?;
            let out = ::std::boxed::Box::new(out);
            let out = ::std::boxed::Box::into_raw(out);
            let out = ::std::ptr::NonNull::new(out as *mut u8).unwrap();
            unsafe { state.write(MaybeUninit::new(StatePtr(out))) };
            Ok(())
        })
    }

    pub unsafe extern "C" fn new_empty<Data: StatefulUdfTrait>(
        data: DataPtr,
        state: StatePtr,
        out: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_ref::<Data::State>() };
            let new_state = data.new_empty(state)?;
            let new_state = ::std::boxed::Box::new(new_state);
            let new_state = ::std::boxed::Box::into_raw(new_state);
            let new_state = ::std::ptr::NonNull::new(new_state as *mut u8).unwrap();
            unsafe { out.write(MaybeUninit::new(StatePtr(new_state))) };
            Ok(())
        })
    }

    pub unsafe extern "C" fn reset<Data: StatefulUdfTrait>(data: DataPtr, state: StatePtr) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            data.reset(state)
        })
    }

    pub unsafe extern "C" fn insert<Data: StatefulUdfTrait>(
        data: DataPtr,
        state: StatePtr,

        inputs_series: *mut SeriesExport,
        inputs_len: usize,

        out_series_is_some: NonNull<bool>,
        out_series: NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            let inputs = unsafe { import_series_buffer(inputs_series, inputs_len) }?;

            let out = data.insert(state, &inputs)?;
            unsafe { out_series_is_some.write(out.is_some()) };
            if let Some(series) = out {
                let exported = export_series(&series);
                unsafe { out_series.write(MaybeUninit::new(exported)) };
            }
            Ok(())
        })
    }

    pub unsafe extern "C" fn finalize<Data: StatefulUdfTrait>(
        data: DataPtr,
        state: StatePtr,
        out_series_is_some: NonNull<bool>,
        out_series: NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            let out = data.finalize(state)?;
            unsafe { out_series_is_some.write(out.is_some()) };
            if let Some(series) = out {
                let exported = export_series(&series);
                unsafe { out_series.write(MaybeUninit::new(exported)) };
            }
            Ok(())
        })
    }

    pub unsafe extern "C" fn combine<Data: StatefulUdfTrait>(
        data: DataPtr,
        state: StatePtr,
        other: StatePtr,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            let other = unsafe { other.as_ref::<Data::State>() };
            data.combine(state, other)
        })
    }

    pub unsafe extern "C" fn to_field<Data: StatefulUdfTrait>(
        data: DataPtr,
        fields: NonNull<ArrowSchema>,
        out: NonNull<MaybeUninit<ArrowSchema>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let fields = unsafe { import_pl_schema(fields) }?;
            let field = data.to_field(&fields)?;
            let field = field.to_arrow(CompatLevel::newest());
            let field = unsafe { export_field_to_c(&field) };
            unsafe { out.write(MaybeUninit::new(field)) };
            Ok(())
        })
    }

    pub unsafe extern "C" fn flags<Data: StatefulUdfTrait>(
        data: DataPtr,
        out: NonNull<u64>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let flags = data.flags().bits();
            unsafe { out.write(flags) };
            Ok(())
        })
    }

    pub unsafe extern "C" fn format<Data: StatefulUdfTrait>(
        data: DataPtr,
        out_ptr: NonNull<*const u8>,
        out_len: NonNull<usize>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let format = data.format();
            unsafe { out_ptr.write(format.as_ptr()) };
            unsafe { out_len.write(format.len()) };
            Ok(())
        })
    }

    pub unsafe extern "C" fn drop_state<Data: StatefulUdfTrait>(state: StatePtr) -> u32 {
        wrap_callee_function(|| {
            let state = unsafe { state.as_ptr::<Data::State>() };
            let state = unsafe { Box::from_raw(state) };
            drop(state);
            Ok(())
        })
    }

    pub unsafe extern "C" fn drop_data<Data: StatefulUdfTrait>(data: DataPtr) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ptr::<Data>() };
            let data = unsafe { Box::from_raw(data) };
            drop(data);
            Ok(())
        })
    }
}

mod _caller {
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;

    use arrow::datatypes::{ArrowDataType, Field as ArrowField};
    use arrow::ffi::ArrowSchema;
    use polars_core::prelude::{CompatLevel, Field, Schema};
    use polars_core::schema::SchemaExt;
    use polars_core::series::Series;
    use polars_error::{PolarsResult, polars_bail};
    use ::polars_ffi::version_0::{SeriesExport, export_series, import_series};
    use polars_utils::pl_str::PlSmallStr;

    use super::super::StatefulUdfTrait;
    use super::{DataPtr, ReturnValue, StatePtr, VTable};
    use crate::dsl::v2::UdfV2Flags;

    impl VTable {
        unsafe fn get_error(&self) -> Option<Box<str>> {
            let mut ptr = std::ptr::null();
            let mut len = 0isize;
            unsafe { (self._get_error)(NonNull::from_mut(&mut ptr), NonNull::from_mut(&mut len)) };
            let Ok(len) = len.try_into() else {
                return None;
            };
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            let s = unsafe { std::str::from_utf8_unchecked(slice) };
            Some(s.into())
        }

        unsafe fn handle_return_value(&self, rv: u32) -> PolarsResult<()> {
            match ReturnValue::from(rv) {
                ReturnValue::Ok => Ok(()),
                ReturnValue::Panic => panic!("plugin panicked"),
                ReturnValue::InvalidOperation => {
                    let msg = unsafe { self.get_error() }.unwrap();
                    polars_bail!(InvalidOperation: "{msg}")
                },
                ReturnValue::ComputeError => {
                    let msg = unsafe { self.get_error() }.unwrap();
                    polars_bail!(ComputeError: "{msg}")
                },
                ReturnValue::ShapeMismatch => {
                    let msg = unsafe { self.get_error() }.unwrap();
                    polars_bail!(ShapeMismatch: "{msg}")
                },
                ReturnValue::AssertionError => {
                    let msg = unsafe { self.get_error() }.unwrap();
                    polars_bail!(AssertionError: "{msg}");
                },
                ReturnValue::OtherError => {
                    let msg = unsafe { self.get_error() }.unwrap();
                    polars_bail!(ComputeError: "unknown error: {msg}")
                },
            }
        }

        unsafe fn handle_return_value_panic(&self, rv: u32) {
            match ReturnValue::from(rv) {
                ReturnValue::Ok => {},
                ReturnValue::Panic => panic!("plugin panicked"),
                _ => panic!("did not expect error"),
            }
        }

        pub unsafe fn initialize(&self, data: DataPtr, fields: &Schema) -> PolarsResult<StatePtr> {
            let mut out_state: MaybeUninit<StatePtr> = MaybeUninit::uninit();
            let fields = arrow::ffi::export_field_to_c(&ArrowField::new(
                PlSmallStr::EMPTY,
                ArrowDataType::Struct(
                    fields
                        .iter_fields()
                        .map(|f| f.to_arrow(CompatLevel::newest()))
                        .collect(),
                ),
                false,
            ));
            let rv = unsafe {
                (self._initialize)(
                    data,
                    NonNull::from_ref(&fields),
                    NonNull::from_mut(&mut out_state),
                )
            };
            unsafe { self.handle_return_value(rv) }?;
            Ok(unsafe { out_state.assume_init() })
        }

        pub unsafe fn new_empty(&self, data: DataPtr, state: StatePtr) -> PolarsResult<StatePtr> {
            let mut out_state: MaybeUninit<StatePtr> = MaybeUninit::uninit();
            let rv = unsafe { (self._new_empty)(data, state, NonNull::from_mut(&mut out_state)) };
            unsafe { self.handle_return_value(rv) }?;
            Ok(unsafe { out_state.assume_init() })
        }

        pub unsafe fn reset(&self, data: DataPtr, state: StatePtr) -> PolarsResult<()> {
            let rv = unsafe { (self._reset)(data, state) };
            unsafe { self.handle_return_value(rv) }?;
            Ok(())
        }

        pub unsafe fn insert(
            &self,
            data: DataPtr,
            state: StatePtr,
            inputs: &[Series],
        ) -> PolarsResult<Option<Series>> {
            let mut inputs_export = Vec::with_capacity(inputs.len());
            for s in inputs {
                inputs_export.push(export_series(s));
            }
            let inputs_ptr = inputs_export.as_mut_ptr();
            let inputs_len = inputs_export.len();

            let mut out_series_is_some = false;
            let mut out_series = MaybeUninit::uninit();

            let rv = unsafe {
                (self._insert)(
                    data,
                    state,
                    inputs_ptr,
                    inputs_len,
                    NonNull::from_mut(&mut out_series_is_some),
                    NonNull::from_mut(&mut out_series),
                )
            };
            unsafe { self.handle_return_value(rv) }?;
            // Already deallocated in insert function
            unsafe { inputs_export.set_len(0) };

            if !out_series_is_some {
                return Ok(None);
            }

            let out_series = unsafe { import_series(out_series.assume_init()) }?;
            Ok(Some(out_series))
        }

        pub unsafe fn finalize(
            &self,
            data: DataPtr,
            state: StatePtr,
        ) -> PolarsResult<Option<Series>> {
            let mut out_series_is_some = false;
            let mut out_series = MaybeUninit::uninit();

            let rv = unsafe {
                (self._finalize)(
                    data,
                    state,
                    NonNull::from_mut(&mut out_series_is_some),
                    NonNull::from_mut(&mut out_series),
                )
            };
            unsafe { self.handle_return_value(rv) }?;

            if !out_series_is_some {
                return Ok(None);
            }

            let out_series = unsafe { import_series(out_series.assume_init()) }?;
            Ok(Some(out_series))
        }

        pub unsafe fn combine(
            &self,
            data: DataPtr,
            state: StatePtr,
            other: StatePtr,
        ) -> PolarsResult<()> {
            let rv = unsafe { (self._combine)(data, state, other) };
            unsafe { self.handle_return_value(rv) }?;
            Ok(())
        }

        pub unsafe fn to_field(&self, data: DataPtr, fields: &Schema) -> PolarsResult<Field> {
            let mut field: MaybeUninit<ArrowSchema> = MaybeUninit::uninit();
            let fields = arrow::ffi::export_field_to_c(&ArrowField::new(
                PlSmallStr::EMPTY,
                ArrowDataType::Struct(
                    fields
                        .iter_fields()
                        .map(|f| f.to_arrow(CompatLevel::newest()))
                        .collect(),
                ),
                false,
            ));
            let rv = unsafe {
                (self._to_field)(
                    data,
                    NonNull::from_ref(&fields),
                    NonNull::from_mut(&mut field),
                )
            };
            unsafe { self.handle_return_value(rv) }?;
            let field = unsafe { field.assume_init() };
            let field = unsafe { arrow::ffi::import_field_from_c(&field) }?;
            Ok(Field::from(&field))
        }

        pub unsafe fn flags(&self, data: DataPtr) -> UdfV2Flags {
            let mut flags = 0;
            let rv = unsafe { (self._flags)(data, NonNull::from_mut(&mut flags)) };
            unsafe { self.handle_return_value_panic(rv) };
            UdfV2Flags::from_bits_truncate(flags)
        }

        pub unsafe fn format(&self, data: DataPtr) -> Box<str> {
            let mut ptr = std::ptr::null();
            let mut len = 0;
            let rv = unsafe {
                (self._format)(
                    data,
                    NonNull::from_mut(&mut ptr),
                    NonNull::from_mut(&mut len),
                )
            };
            unsafe { self.handle_return_value_panic(rv) };
            let slice = unsafe { std::slice::from_raw_parts(ptr, len) };
            let s = unsafe { std::str::from_utf8_unchecked(slice) };
            s.into()
        }

        pub unsafe fn drop_state(&self, state: StatePtr) {
            let rv = unsafe { (self._drop_state)(state) };
            unsafe { self.handle_return_value_panic(rv) };
        }

        pub unsafe fn drop_data(&self, data: DataPtr) {
            let rv = unsafe { (self._drop_data)(data) };
            unsafe { self.handle_return_value_panic(rv) };
        }
    }
}
