use std::borrow::Cow;
use std::mem::MaybeUninit;
use std::ptr::NonNull;

use arrow::ffi::ArrowSchema;
use polars_core::prelude::Field;
use polars_core::schema::Schema;
use polars_core::series::Series;
use polars_error::PolarsResult;

use crate::version_0::SeriesExport;

/// Symbol exposed in shared library to Polars.
#[repr(C, align(8))]
pub struct PluginSymbol {
    pub version: u64,
    pub vtable: VTable,
}
/// Version of Plugin V1.
pub const VERSION: u32 = 0x00_00_01;

/// Pointer to a `Box<T>` where `T: PolarsPlugin`.
#[repr(transparent)]
pub struct DataPtr(NonNull<u8>);
/// Pointer to a `Box<T::State>` where `T: PolarsPlugin`.
#[repr(transparent)]
pub struct StatePtr(NonNull<u8>);

// pub trait ElementwisePlugin: Sereajfdlakjf + Dserajlkj {
//     fn to_field() -> ...;
//     fn evaluate(inputs: &[Series]) -> PolarsResult<Series>;
// }
//
// pub trait MapReducePlugin: Sereajfdlakjf + Dserajlkj {
//     fn to_field() -> ...;
//
//     fn insert(&self, inputs: &[Series]) -> PolarsResult<Self::State>;
//     fn combine(&self, left: Self::State, right: Self::State) -> PolarsResult<Self::State>;
//     fn finalize(&self, state: Self::State) -> PolarsResult<Series>;
// }
//
// pub trait ScanPlugin: Sereajfdlakjf + Dserajlkj {
//     fn to_field() -> PolarsResult<Field>;
//
//     fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Series>;
//     fn finalize(&self, state: Self::State) -> PolarsResult<Option<Series>>;
// }
//
pub trait PolarsPlugin: Send + Sync + Sized {
    type State: Send + Sync + Sized;

    // (De)serialization methods.
    /// Serialize `Self` into a buffer.
    fn serialize(&self) -> PolarsResult<Box<[u8]>>;

    /// Deserialize `Self` from a buffer.
    fn deserialize(buffer: &[u8]) -> PolarsResult<Self>;

    /// Serialize `Self::State` into a buffer.
    fn serialize_state(&self, state: &Self::State) -> PolarsResult<Box<[u8]>>;

    /// Deserialize `Self::State` from a buffer.
    fn deserialize_state(&self, buffer: &[u8]) -> PolarsResult<Self::State>;

    // Planning methods.
    /// Get the output field for some given input `fields`.
    fn to_field(&self, fields: &Schema) -> PolarsResult<Field>;

    // State management methods.
    /// Create a new state for specific input `fields`.
    fn new_state(&self, fields: &Schema) -> PolarsResult<Self::State>;
    /// Clone and reset a state.
    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State>;
    /// Reset a state and prepare for receiving a new `step` calls.
    fn reset(&self, state: &mut Self::State) -> PolarsResult<()>;
    /// Fold the `other` state into `state`.
    ///
    /// This is never called if the states are not combinable.
    fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()>;

    // Execution methods.
    /// Evaluate the function for a part of the input data, updating the state and possibly
    /// returning some output data.
    ///
    /// Invariants:
    /// * This may be called any number of times and should yield an equally valid result given the
    ///   same `self`, `state` and `inputs`.
    /// * If the inputs are zippable, this is always called with equal length data for `non-scalar`
    ///   inputs.
    /// * Subsequent calls to `step` with the same `state` never provide the data of the input
    ///   expressions out-of-order unless the expression is explicitly set to not observe the input
    ///   order.
    /// * If states are **not** combinable, the combination of all `step` calls will receive all
    ///   input data.
    /// * If step is said to have no output, the result is always to be either `Err` or `Ok(None)`.
    fn step(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>>;

    /// Finalize a state and optionally return the final output.
    ///
    /// This indicates that this state will not `step` anymore.
    fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>>;

    fn evaluate_on_groups<'a>(
        &self,
        inputs: &[(Series, &'a GroupPositions)],
    ) -> PolarsResult<(Series, Cow<'a, GroupPositions>)> {
        _ = inputs;
        dbg!();
        todo!()
    }
}

#[repr(C)]
#[derive(Clone, Copy)]
pub struct SliceGroup {
    pub offset: u64,
    pub length: u64,
}

#[repr(C)]
#[derive(Clone)]
pub struct IndexGroups {
    pub index: Box<[u64]>,
    pub ends: Box<[u64]>,
}

#[repr(C)]
#[derive(Clone)]
pub enum GroupPositions {
    /// Every group has all the data values.
    SharedAcrossGroups,
    /// Every group has 1 value in sequential order of the data.
    ScalarPerGroup,
    Slice(Box<[SliceGroup]>),
    Index(IndexGroups),
}

pub struct CowGroupPositions<'a> {
    groups: &'a GroupPositions,
    drop: Option<unsafe extern "C" fn(*mut GroupPositions) -> u32>,
}

impl<'a> AsRef<GroupPositions> for CowGroupPositions<'a> {
    fn as_ref(&self) -> &GroupPositions {
        self.groups
    }
}

impl<'a> Drop for CowGroupPositions<'a> {
    fn drop(&mut self) {
        if let Some(drop) = &self.drop {
            let rv = unsafe { (drop)(self.groups as *const GroupPositions as *mut GroupPositions) };
            match ReturnValue::from(rv) {
                ReturnValue::Ok => {},
                ReturnValue::Panic => panic!("plugin panicked"),
                _ => panic!("did not expect error"),
            }
        }
    }
}

impl IndexGroups {
    pub fn lengths(&self) -> impl Iterator<Item = usize> {
        (0..self.ends.len()).map(|i| {
            let start = i.checked_sub(1).map_or(0, |i| self.ends[i]);
            let end = self.ends[i];
            (end - start) as usize
        })
    }

    pub fn iter(&self) -> impl Iterator<Item = &[u64]> {
        (0..self.ends.len()).map(|i| {
            let start = i.checked_sub(1).map_or(0, |i| self.ends[i]);
            let end = self.ends[i];

            &self.index[start as usize..end as usize]
        })
    }
}

#[derive(Clone)]
#[repr(C)]
pub struct VTable {
    /// Serialize `Data` into a byte buffer.
    ///
    /// The callee allocates a `Box<[u8]>` and writes the pointer and length into `buffer` and
    /// `length`. This `Box<[u8]>` is then later dropped using `_drop_box_byte_slice`.
    ///
    /// The `buffer` pointer may be NULL i.f.f. `length` is 0.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data` should be valid for the entire duration.
    /// - `buffer` should point to a valid and exclusive referenceable memory location.
    /// - `length` should point to a valid and exclusive referenceable memory location.
    _serialize_data: unsafe extern "C" fn(
        data: DataPtr,            // Read-only reference to the data.
        buffer: NonNull<*mut u8>, // Pointer to a byte buffer pointer.
        length: NonNull<usize>,   // Pointer to a length buffer.
    ) -> u32,
    /// Deserialize `Data` from a byte buffer.
    ///
    /// The callee allocates a `Box<Data>` and writes the pointer to `data`. This data is then
    /// later dropped using `_drop_data`.
    ///
    /// The `buffer` pointer may be NULL i.f.f. `length` is 0.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - `buffer` and `length` should be valid raw parts of a `&[u8]`.
    /// - `data` should point to a valid and exclusive referenceable memory location.
    _deserialize_data: unsafe extern "C" fn(
        buffer: *const u8,
        length: usize,
        data: NonNull<MaybeUninit<DataPtr>>,
    ) -> u32,
    /// Serialize `state` into a byte buffer.
    ///
    /// The callee allocates a `Box<[u8]>` and writes the pointer and length into `buffer` and
    /// `length`. This `Box<[u8]>` is then later dropped using `_drop_box_byte_slice`.
    ///
    /// The `buffer` pointer may be NULL i.f.f. `length` is 0.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data` and `state` should be valid for the entire duration.
    /// - `buffer` should point to a valid and exclusive referenceable memory location.
    /// - `length` should point to a valid and exclusive referenceable memory location.
    _serialize_state: unsafe extern "C" fn(
        data: DataPtr,
        state: StatePtr,
        buffer: NonNull<*mut u8>,
        length: NonNull<usize>,
    ) -> u32,
    /// Deserialize `State` from a byte buffer.
    ///
    /// The callee allocates a `Box<State>` and writes the pointer to `state`. This state is then
    /// later dropped using `_drop_state`.
    ///
    /// The `buffer` pointer may be NULL i.f.f. `length` is 0.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data` should be valid for the entire duration.
    /// - `buffer` and `length` should be valid raw parts of a `&[u8]`.
    /// - `state` should point to a valid and exclusive referenceable memory location.
    _deserialize_state: unsafe extern "C" fn(
        data: DataPtr,
        buffer: *const u8,
        length: usize,
        state: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32,
    /// Drop `Box<[u8]>`.
    ///
    /// Returns a `ReturnValue` to handle panics.
    ///
    /// # Safety
    ///
    /// - `ptr` and `length` should be the components belonging to a `Box<[u8]>` on the callee's allocator.
    _drop_box_byte_slice: unsafe extern "C" fn(ptr: *mut u8, length: usize) -> u32,

    /// Allocate and initialize a new `Box<State>` and put its pointer in `state`.
    ///
    /// This state is then later dropped using `_drop_state`.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data` and `fields` should be valid for the entire duration.
    /// - `state` should point to a valid and exclusive referenceable memory location.
    _new_state: unsafe extern "C" fn(
        data: DataPtr,
        fields: NonNull<ArrowSchema>,
        state: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32,
    /// Clone `state`, reset and put a newly allocated `Box<State>`'s pointer into `new`.
    ///
    /// This state is then later dropped using `_drop_state`.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data` and `state` should be valid for the entire duration.
    /// - `new` should point to a valid and exclusive referenceable memory location.
    _new_empty: unsafe extern "C" fn(
        data: DataPtr,
        state: StatePtr,
        new: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32,
    /// Reset `state`.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data` and `state` should be valid for the entire duration.
    _reset: unsafe extern "C" fn(data: DataPtr, state: StatePtr) -> u32,
    /// Fold `other` into `state`.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// # Safety
    ///
    /// - Memory backing `data`, `state` and `other` should be valid for the entire duration.
    /// - `state` should be exclusively referenceable
    _combine: unsafe extern "C" fn(data: DataPtr, state: StatePtr, other: StatePtr) -> u32,

    /// Evaluate one execution step.
    ///
    /// The caller passes the ownership of all elements in `inputs` to the caller, but not the
    /// slice itself.
    ///
    /// - `out_kind` should be `Value`.
    /// - `out_series` will be initialized i.f.f. `out_kind` is `Value::Series`.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// - Memory backing `data` and `state` should be valid for the entire duration.
    /// - `inputs_ptr` and `inputs_len` should represent owned contiguous `SeriesExport`.
    /// - `out_kind` and `out_series should be exclusively referenceable
    _step: unsafe extern "C" fn(
        data: DataPtr,
        state: StatePtr,
        inputs_ptr: *mut SeriesExport,
        inputs_len: usize,
        out_kind: NonNull<u32>,
        out_series: NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32,
    /// Finalize a state.
    ///
    /// - `out_kind` should be `Value`.
    /// - `out_series` will be initialized i.f.f. `out_kind` is `Value::Series`.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// - Memory backing `data` and `state` should be valid for the entire duration.
    /// - `out_kind` and `out_series should be exclusively referenceable
    _finalize: unsafe extern "C" fn(
        data: DataPtr,
        state: StatePtr,
        out_kind: NonNull<u32>,
        out_series: NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32,

    _evaluate_on_groups: unsafe extern "C" fn(
        data: DataPtr,
        inputs_ptr: *mut (SeriesExport, NonNull<GroupPositions>),
        inputs_len: usize,
        output_series: NonNull<MaybeUninit<SeriesExport>>,
        output_groups_owned: NonNull<MaybeUninit<bool>>,
        output_groups: NonNull<MaybeUninit<NonNull<GroupPositions>>>,
    ) -> u32,
    _drop_box_group_positions: unsafe extern "C" fn(ptr: *mut GroupPositions) -> u32,

    /// Get the output field.
    ///
    /// Returns a `ReturnValue` to handle panics and errors.
    ///
    /// - Memory backing `data` and `fields` should be valid for the entire duration.
    /// - `out_field` should be exclusively referenceable
    _to_field: unsafe extern "C" fn(
        data: DataPtr,
        fields: NonNull<ArrowSchema>,
        out_field: NonNull<MaybeUninit<ArrowSchema>>,
    ) -> u32,

    /// Drop `Box<Data>`.
    ///
    /// Returns a `ReturnValue` to handle panics.
    ///
    /// # Safety
    ///
    /// - `ptr` and `length` should be the components belonging to a `Box<Data>` on the callee's allocator.
    _drop_state: unsafe extern "C" fn(StatePtr) -> u32,
    /// Drop `Box<State>`.
    ///
    /// Returns a `ReturnValue` to handle panics.
    ///
    /// # Safety
    ///
    /// - `ptr` and `length` should be the components belonging to a `Box<State>` on the callee's allocator.
    _drop_data: unsafe extern "C" fn(DataPtr) -> u32,

    /// Set the `caller` plugin v1 version in the callee.
    _set_version: unsafe extern "C" fn(u32),
    /// Get a error message of the last error that occurred.
    _get_error: unsafe extern "C" fn(NonNull<*const u8>, NonNull<isize>),
}

/// Value that is returned from a VTable function call.
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

/// Kind of value that was returned from `step` or `finalize`.
#[repr(u32)]
enum Value {
    None = 0,
    Series = 1,
}

impl DataPtr {
    #[doc(hidden)]
    pub fn _new(ptr: NonNull<u8>) -> Self {
        Self(ptr)
    }

    /// # Safety
    ///
    /// Shared access and not yet free-ed.
    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { self.0.as_ptr().cast::<T>().as_ref() }.unwrap()
    }

    /// # Safety
    ///
    /// Exclusive access and not yet free-ed.
    #[expect(clippy::mut_from_ref)]
    pub unsafe fn as_mut<T>(&self) -> &mut T {
        unsafe { self.0.as_ptr().cast::<T>().as_mut() }.unwrap()
    }

    /// # Safety
    ///
    /// Underlying Data is not free-ed before the result is dropped.
    pub unsafe fn ptr_clone(&self) -> Self {
        Self(self.0)
    }
}

impl StatePtr {
    /// # Safety
    ///
    /// Shared access and not yet free-ed.
    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { self.0.as_ptr().cast::<T>().as_ref() }.unwrap()
    }

    /// # Safety
    ///
    /// Exclusive access and not yet free-ed.
    #[expect(clippy::mut_from_ref)]
    pub unsafe fn as_mut<T>(&self) -> &mut T {
        unsafe { self.0.as_ptr().cast::<T>().as_mut() }.unwrap()
    }

    /// # Safety
    ///
    /// Underlying Data is not free-ed before the result is dropped.
    pub unsafe fn ptr_clone(&self) -> Self {
        Self(self.0)
    }
}

impl TryFrom<u32> for Value {
    type Error = ();
    fn try_from(value: u32) -> Result<Self, Self::Error> {
        match value {
            0 => Ok(Self::None),
            1 => Ok(Self::Series),
            _ => Err(()),
        }
    }
}

impl VTable {
    pub const fn new<Data: PolarsPlugin>() -> Self {
        Self {
            _serialize_data: _callee::serialize_data::<Data>,
            _deserialize_data: _callee::deserialize_data::<Data>,
            _serialize_state: _callee::serialize_state::<Data>,
            _deserialize_state: _callee::deserialize_state::<Data>,
            _drop_box_byte_slice: _callee::drop_box_byte_slice,

            _new_state: _callee::new_state::<Data>,
            _new_empty: _callee::new_empty::<Data>,
            _reset: _callee::reset::<Data>,

            _step: _callee::step::<Data>,
            _finalize: _callee::finalize::<Data>,
            _combine: _callee::combine::<Data>,

            _evaluate_on_groups: _callee::evaluate_on_groups::<Data>,
            _drop_box_group_positions: _callee::drop_box_group_positions,

            _to_field: _callee::to_field::<Data>,
            _drop_state: _callee::drop_state::<Data>,
            _drop_data: _callee::drop_data::<Data>,

            _set_version: _callee::set_version,
            _get_error: _callee::get_error,
        }
    }

    pub const fn into_symbol(self) -> PluginSymbol {
        PluginSymbol {
            version: VERSION as u64,
            vtable: self,
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

/// FFI Wrappers for all [`PolarsPlugin`] functions from the *callee* or Plugin's side.
mod _callee {
    use std::borrow::Cow;
    use std::cell::RefCell;
    use std::mem::MaybeUninit;
    use std::panic::UnwindSafe;
    use std::ptr::NonNull;
    use std::sync::atomic::{AtomicU32, Ordering};

    use arrow::datatypes::ArrowDataType;
    use arrow::ffi::{ArrowSchema, export_field_to_c, import_field_from_c};
    use polars_core::prelude::{CompatLevel, Schema};
    use polars_core::schema::SchemaExt;
    use polars_error::{PolarsError, PolarsResult, polars_bail};

    use super::{DataPtr, GroupPositions, PolarsPlugin, ReturnValue, StatePtr, Value};
    use crate::version_0::{SeriesExport, export_series, import_series, import_series_buffer};

    /// Plugin version of the *caller*.
    ///
    /// This can be used to assess whether certain features are usable or not.
    static POLARS_PLUGIN_VERSION: AtomicU32 = AtomicU32::new(0);
    thread_local! { static ERROR_MESSAGE: RefCell<Option<Box<str>>> = const { RefCell::new(None) }; }

    /// # Safety
    ///
    /// See VTable.
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

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn set_version(version: u32) {
        POLARS_PLUGIN_VERSION.store(version, Ordering::Relaxed);
    }

    /// # Safety
    ///
    /// Valid schema.
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
        let result = std::panic::catch_unwind(f);

        let set_msg = match &result {
            Ok(Ok(_)) => Ok(()),
            Err(_) => Ok(()),
            Ok(Err(err)) => {
                let (kind, msg) = match err {
                    PolarsError::InvalidOperation(msg)
                    | PolarsError::ComputeError(msg)
                    | PolarsError::ShapeMismatch(msg)
                    | PolarsError::AssertionError(msg) => (None, Some(msg)),
                    PolarsError::Duplicate(msg) => (Some("DuplicateError"), Some(msg)),
                    PolarsError::IO { error: _, msg: _ } => (Some("IO"), None),
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
                    PolarsError::Context { error: _, msg: _ } => (Some("unknown"), None),
                    #[cfg(feature = "python")]
                    PolarsError::Python { error: _ } => (Some("python"), None),
                };

                let msg = msg.map(|m| m.as_ref());
                match (kind, msg) {
                    (None, None) => Ok(()),
                    (Some(m), None) | (None, Some(m)) => std::panic::catch_unwind(move || {
                        ERROR_MESSAGE.with_borrow_mut(|b| *b = Some(m.into()))
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

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn serialize_data<Data: PolarsPlugin>(
        data: DataPtr,
        buffer: NonNull<*mut u8>,
        length: NonNull<usize>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let out = data.serialize()?;
            let out = Box::leak(out);
            unsafe { buffer.write(out.as_mut_ptr()) };
            unsafe { length.write(out.len()) };
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn deserialize_data<Data: PolarsPlugin>(
        buffer: *const u8,
        length: usize,
        data: NonNull<MaybeUninit<DataPtr>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let buffer = unsafe { std::slice::from_raw_parts(buffer, length) };
            let out = Data::deserialize(buffer)?;
            let out = Box::new(out);
            let out = Box::into_raw(out);
            let out = NonNull::new(out as *mut u8).unwrap();
            unsafe { data.write(MaybeUninit::new(DataPtr(out))) };
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn serialize_state<Data: PolarsPlugin>(
        data: DataPtr,
        state: StatePtr,
        buffer: NonNull<*mut u8>,
        length: NonNull<usize>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_ref::<Data::State>() };
            let out = data.serialize_state(state)?;
            let out = Box::leak(out);
            unsafe { buffer.write(out.as_mut_ptr()) };
            unsafe { length.write(out.len()) };
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn deserialize_state<Data: PolarsPlugin>(
        data: DataPtr,
        buffer: *const u8,
        length: usize,
        state: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let buffer = unsafe { std::slice::from_raw_parts(buffer, length) };
            let out = data.deserialize_state(buffer)?;
            let out = Box::new(out);
            let out = Box::into_raw(out);
            let out = NonNull::new(out as *mut u8).unwrap();
            unsafe { state.write(MaybeUninit::new(StatePtr(out))) };
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn drop_box_byte_slice(buffer: *mut u8, length: usize) -> u32 {
        wrap_callee_function(|| {
            let buffer = unsafe { std::slice::from_raw_parts_mut(buffer, length) };
            let buffer = unsafe { Box::from_raw(buffer as *mut [u8]) };
            drop(buffer);
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn new_state<Data: PolarsPlugin>(
        data: DataPtr,
        fields: NonNull<ArrowSchema>,
        state: NonNull<MaybeUninit<StatePtr>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let fields = unsafe { import_pl_schema(fields) }?;
            let out = data.new_state(&fields)?;
            let out = ::std::boxed::Box::new(out);
            let out = ::std::boxed::Box::into_raw(out);
            let out = ::std::ptr::NonNull::new(out as *mut u8).unwrap();
            unsafe { state.write(MaybeUninit::new(StatePtr(out))) };
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn new_empty<Data: PolarsPlugin>(
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

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn reset<Data: PolarsPlugin>(data: DataPtr, state: StatePtr) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            data.reset(state)
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn step<Data: PolarsPlugin>(
        data: DataPtr,
        state: StatePtr,

        inputs_series: *mut SeriesExport,
        inputs_len: usize,

        out_kind: NonNull<u32>,
        out_series: NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            let inputs = unsafe { import_series_buffer(inputs_series, inputs_len) }?;

            let out = data.step(state, &inputs)?;
            let kind = match out {
                None => Value::None as u32,
                Some(_) => Value::Series as u32,
            };
            unsafe { out_kind.write(kind) };
            if let Some(series) = out {
                let exported = export_series(&series);
                unsafe { out_series.write(MaybeUninit::new(exported)) };
            }
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn finalize<Data: PolarsPlugin>(
        data: DataPtr,
        state: StatePtr,
        out_kind: NonNull<u32>,
        out_series: NonNull<MaybeUninit<SeriesExport>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let state = unsafe { state.as_mut::<Data::State>() };
            let out = data.finalize(state)?;
            let kind = match out {
                None => Value::None as u32,
                Some(_) => Value::Series as u32,
            };
            unsafe { out_kind.write(kind) };
            if let Some(series) = out {
                let exported = export_series(&series);
                unsafe { out_series.write(MaybeUninit::new(exported)) };
            }
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn evaluate_on_groups<Data: PolarsPlugin>(
        data: DataPtr,
        inputs_ptr: *mut (SeriesExport, NonNull<GroupPositions>),
        inputs_len: usize,
        output_series: NonNull<MaybeUninit<SeriesExport>>,
        output_groups_owned: NonNull<MaybeUninit<bool>>,
        output_groups: NonNull<MaybeUninit<NonNull<GroupPositions>>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let mut collected_inputs = Vec::with_capacity(inputs_len);
            for i in 0..inputs_len {
                let (series, groups) = unsafe { std::ptr::read(inputs_ptr.add(i)) };
                let series = unsafe { import_series(series)? };
                let groups = unsafe { groups.as_ref() };
                collected_inputs.push((series, groups));
            }
            let (out_data, out_groups) = data.evaluate_on_groups(&collected_inputs)?;

            let is_owned = matches!(out_groups, Cow::Owned(_));
            unsafe { output_groups_owned.write(MaybeUninit::new(is_owned)) };
            let out_groups = match out_groups {
                Cow::Borrowed(ptr) => NonNull::from_ref(ptr),
                Cow::Owned(out_groups) => {
                    NonNull::new(Box::into_raw(Box::new(out_groups))).unwrap()
                },
            };
            unsafe {
                output_groups.write(MaybeUninit::new(out_groups));
            }
            let out_data = export_series(&out_data);
            unsafe { output_series.write(MaybeUninit::new(out_data)) };

            Ok(())
        })
    }

    pub unsafe extern "C" fn drop_box_group_positions(ptr: *mut GroupPositions) -> u32 {
        wrap_callee_function(|| {
            let buffer = unsafe { Box::from_raw(ptr) };
            drop(buffer);
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn combine<Data: PolarsPlugin>(
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

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn to_field<Data: PolarsPlugin>(
        data: DataPtr,
        fields: NonNull<ArrowSchema>,
        out: NonNull<MaybeUninit<ArrowSchema>>,
    ) -> u32 {
        wrap_callee_function(|| {
            let data = unsafe { data.as_ref::<Data>() };
            let fields = unsafe { import_pl_schema(fields) }?;
            let field = data.to_field(&fields)?;
            let field = field.to_arrow(CompatLevel::newest());
            let field = export_field_to_c(&field);
            unsafe { out.write(MaybeUninit::new(field)) };
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn drop_state<Data: PolarsPlugin>(state: StatePtr) -> u32 {
        wrap_callee_function(|| {
            let state = state.0.as_ptr().cast::<Data::State>();
            let state = unsafe { Box::from_raw(state) };
            drop(state);
            Ok(())
        })
    }

    /// # Safety
    ///
    /// See VTable.
    pub unsafe extern "C" fn drop_data<Data: PolarsPlugin>(data: DataPtr) -> u32 {
        wrap_callee_function(|| {
            let data = data.0.as_ptr().cast::<Data>();
            let data = unsafe { Box::from_raw(data) };
            drop(data);
            Ok(())
        })
    }
}

/// FFI Wrappers for all [`PolarsPlugin`] functions from the *caller* or Polars' side.
mod _caller {
    use std::mem::MaybeUninit;
    use std::ptr::NonNull;

    use arrow::datatypes::{ArrowDataType, Field as ArrowField};
    use arrow::ffi::ArrowSchema;
    use polars_core::prelude::{CompatLevel, Field, Schema};
    use polars_core::schema::SchemaExt;
    use polars_core::series::Series;
    use polars_error::{PolarsResult, polars_bail};
    use polars_utils::pl_str::PlSmallStr;

    use super::{CowGroupPositions, DataPtr, GroupPositions, ReturnValue, StatePtr, VTable, Value};
    use crate::version_0::{export_series, import_series};

    impl VTable {
        /// # Safety
        ///
        /// VTable is filled from callee::get_error.
        fn get_error(&self) -> Option<Box<str>> {
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

        /// # Safety
        ///
        /// VTable is filled from callee::set_version.
        pub fn set_version(&self, version: u32) {
            unsafe { (self._set_version)(version) }
        }

        fn handle_return_value(&self, rv: u32) -> PolarsResult<()> {
            match ReturnValue::from(rv) {
                ReturnValue::Ok => Ok(()),
                ReturnValue::Panic => {
                    panic!("plugin panicked")
                },
                ReturnValue::InvalidOperation => {
                    let msg = self.get_error().unwrap();
                    polars_bail!(InvalidOperation: "{msg}")
                },
                ReturnValue::ComputeError => {
                    let msg = self.get_error().unwrap();
                    polars_bail!(ComputeError: "{msg}")
                },
                ReturnValue::ShapeMismatch => {
                    let msg = self.get_error().unwrap();
                    polars_bail!(ShapeMismatch: "{msg}")
                },
                ReturnValue::AssertionError => {
                    let msg = self.get_error().unwrap();
                    polars_bail!(AssertionError: "{msg}");
                },
                ReturnValue::OtherError => {
                    let msg = self.get_error().unwrap();
                    polars_bail!(ComputeError: "unknown error: {msg}")
                },
            }
        }

        fn handle_return_value_panic(&self, rv: u32) {
            match ReturnValue::from(rv) {
                ReturnValue::Ok => {},
                ReturnValue::Panic => panic!("plugin panicked"),
                _ => panic!("did not expect error"),
            }
        }

        /// # Safety
        ///
        /// `data` is valid and belonging to this VTable.
        pub unsafe fn serialize_data(&self, data: DataPtr, out: &mut Vec<u8>) -> PolarsResult<()> {
            let mut buffer = std::ptr::null_mut();
            let mut length = 0;

            let rv = unsafe {
                (self._serialize_data)(
                    data,
                    NonNull::from_mut(&mut buffer),
                    NonNull::from_mut(&mut length),
                )
            };
            self.handle_return_value(rv)?;

            let slice = unsafe { std::slice::from_raw_parts(buffer, length) };
            out.extend_from_slice(slice);

            let rv = unsafe { (self._drop_box_byte_slice)(buffer, length) };
            self.handle_return_value_panic(rv);
            Ok(())
        }

        pub fn deserialize_data(&self, buffer: &[u8]) -> PolarsResult<DataPtr> {
            let mut data = MaybeUninit::uninit();

            let rv = unsafe {
                (self._deserialize_data)(
                    buffer.as_ptr(),
                    buffer.len(),
                    NonNull::from_mut(&mut data),
                )
            };
            self.handle_return_value(rv)?;

            Ok(unsafe { data.assume_init() })
        }

        /// # Safety
        ///
        /// `data` and `state` are valid and belonging to this VTable.
        pub unsafe fn serialize_state(
            &self,
            data: DataPtr,
            state: StatePtr,
            out: &mut Vec<u8>,
        ) -> PolarsResult<()> {
            let mut buffer = std::ptr::null_mut();
            let mut length = 0;

            let rv = unsafe {
                (self._serialize_state)(
                    data,
                    state,
                    NonNull::from_mut(&mut buffer),
                    NonNull::from_mut(&mut length),
                )
            };
            self.handle_return_value(rv)?;

            let slice = unsafe { std::slice::from_raw_parts(buffer, length) };
            out.extend_from_slice(slice);

            let rv = unsafe { (self._drop_box_byte_slice)(buffer, length) };
            self.handle_return_value_panic(rv);
            Ok(())
        }

        /// # Safety
        ///
        /// `data` is valid and belonging to this VTable.
        pub unsafe fn deserialize_state(
            &self,
            data: DataPtr,
            buffer: &[u8],
        ) -> PolarsResult<StatePtr> {
            let mut state = MaybeUninit::uninit();

            let rv = unsafe {
                (self._deserialize_state)(
                    data,
                    buffer.as_ptr(),
                    buffer.len(),
                    NonNull::from_mut(&mut state),
                )
            };
            self.handle_return_value(rv)?;

            Ok(unsafe { state.assume_init() })
        }

        /// # Safety
        ///
        /// `data` is valid and belonging to this VTable.
        pub unsafe fn new_state(&self, data: DataPtr, fields: &Schema) -> PolarsResult<StatePtr> {
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
                (self._new_state)(
                    data,
                    NonNull::from_ref(&fields),
                    NonNull::from_mut(&mut out_state),
                )
            };
            self.handle_return_value(rv)?;
            Ok(unsafe { out_state.assume_init() })
        }

        /// # Safety
        ///
        /// `data` and `state` are valid and belonging to this VTable.
        pub unsafe fn new_empty(&self, data: DataPtr, state: StatePtr) -> PolarsResult<StatePtr> {
            let mut out_state: MaybeUninit<StatePtr> = MaybeUninit::uninit();
            let rv = unsafe { (self._new_empty)(data, state, NonNull::from_mut(&mut out_state)) };
            self.handle_return_value(rv)?;
            Ok(unsafe { out_state.assume_init() })
        }

        /// # Safety
        ///
        /// `data` and `state` are valid and belonging to this VTable.
        pub fn reset(&self, data: DataPtr, state: StatePtr) -> PolarsResult<()> {
            let rv = unsafe { (self._reset)(data, state) };
            self.handle_return_value(rv)?;
            Ok(())
        }

        /// # Safety
        ///
        /// `data` and `state` are valid and belonging to this VTable.
        pub unsafe fn step(
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

            let mut out_kind = 0u32;
            let mut out_series = MaybeUninit::uninit();

            let rv = unsafe {
                (self._step)(
                    data,
                    state,
                    inputs_ptr,
                    inputs_len,
                    NonNull::from_mut(&mut out_kind),
                    NonNull::from_mut(&mut out_series),
                )
            };
            // Already deallocated in step function
            unsafe { inputs_export.set_len(0) };
            self.handle_return_value(rv)?;

            let Ok(out_kind) = Value::try_from(out_kind) else {
                panic!("invalid series kind value");
            };
            match out_kind {
                Value::None => Ok(None),
                Value::Series => {
                    let out_series = unsafe { import_series(out_series.assume_init()) }?;
                    Ok(Some(out_series))
                },
            }
        }

        /// # Safety
        ///
        /// `data` and `state` are valid and belonging to this VTable.
        pub unsafe fn finalize(
            &self,
            data: DataPtr,
            state: StatePtr,
        ) -> PolarsResult<Option<Series>> {
            let mut out_kind = 0u32;
            let mut out_series = MaybeUninit::uninit();

            let rv = unsafe {
                (self._finalize)(
                    data,
                    state,
                    NonNull::from_mut(&mut out_kind),
                    NonNull::from_mut(&mut out_series),
                )
            };
            self.handle_return_value(rv)?;

            let Ok(out_kind) = Value::try_from(out_kind) else {
                panic!("invalid series kind value");
            };
            match out_kind {
                Value::None => Ok(None),
                Value::Series => {
                    let out_series = unsafe { import_series(out_series.assume_init()) }?;
                    Ok(Some(out_series))
                },
            }
        }

        /// # Safety
        ///
        /// `data` and `state` are valid and belonging to this VTable.
        pub unsafe fn combine(
            &self,
            data: DataPtr,
            state: StatePtr,
            other: StatePtr,
        ) -> PolarsResult<()> {
            let rv = unsafe { (self._combine)(data, state, other) };
            self.handle_return_value(rv)?;
            Ok(())
        }

        /// # Safety
        ///
        /// `data` is valid and belonging to this VTable.
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
            self.handle_return_value(rv)?;
            let field = unsafe { field.assume_init() };
            let field = unsafe { arrow::ffi::import_field_from_c(&field) }?;
            Ok(Field::from(&field))
        }

        pub unsafe fn evaluate_on_groups<'a>(
            &self,
            data: DataPtr,
            inputs: &[(Series, &'a GroupPositions)],
        ) -> PolarsResult<(Series, CowGroupPositions<'a>)> {
            let mut out_series = MaybeUninit::uninit();
            let mut out_groups_owned = MaybeUninit::uninit();
            let mut out_groups = MaybeUninit::uninit();

            let mut inputs_export = Vec::with_capacity(inputs.len());
            for (series, groups) in inputs {
                let series = export_series(series);
                let groups = NonNull::from_ref(*groups);
                inputs_export.push((series, groups));
            }

            let rv = unsafe {
                (self._evaluate_on_groups)(
                    data,
                    inputs_export.as_mut_ptr(),
                    inputs.len(),
                    NonNull::from_mut(&mut out_series),
                    NonNull::from_mut(&mut out_groups_owned),
                    NonNull::from_mut(&mut out_groups),
                )
            };
            // Already deallocated in step function
            unsafe { inputs_export.set_len(0) };
            self.handle_return_value(rv)?;

            let out_series = unsafe { out_series.assume_init() };
            let out_series = unsafe { import_series(out_series) }?;
            let out_groups_owned = unsafe { out_groups_owned.assume_init() };
            let out_groups = unsafe { out_groups.assume_init() };
            let drop_fn = out_groups_owned.then_some(self._drop_box_group_positions);
            Ok((
                out_series,
                CowGroupPositions {
                    groups: unsafe { out_groups.as_ref() },
                    drop: drop_fn,
                },
            ))
        }

        /// # Safety
        ///
        /// `state` is valid and belonging to this VTable.
        pub unsafe fn drop_state(&self, state: StatePtr) {
            let rv = unsafe { (self._drop_state)(state) };
            self.handle_return_value_panic(rv);
        }

        /// # Safety
        ///
        /// `data` is valid and belonging to this VTable.
        pub unsafe fn drop_data(&self, data: DataPtr) {
            let rv = unsafe { (self._drop_data)(data) };
            self.handle_return_value_panic(rv);
        }
    }
}
