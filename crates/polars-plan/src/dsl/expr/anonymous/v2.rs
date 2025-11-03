use core::str;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use arrow::ffi::ArrowSchema;
use polars_core::frame::DataFrame;
use polars_core::prelude::{CompatLevel, Field};
use polars_core::schema::{Schema, SchemaExt};
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail};
use polars_ffi::version_0::SeriesExport;
use polars_utils::pl_str::PlSmallStr;

#[repr(transparent)]
pub struct DataPtr(NonNull<u8>);
#[repr(transparent)]
pub struct StatePtr(NonNull<u8>);

impl DataPtr {
    #[doc(hidden)]
    pub unsafe fn _new(ptr: NonNull<u8>) -> Self {
        Self(ptr)
    }

    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { self.0.as_ptr().cast::<T>().as_ref() }.unwrap()
    }

    pub unsafe fn as_mut<T>(&self) -> &mut T {
        unsafe { self.0.as_ptr().cast::<T>().as_mut() }.unwrap()
    }

    pub unsafe fn as_ptr<T>(&self) -> *mut T {
        unsafe { self.0.as_ptr().cast::<T>() }
    }

    unsafe fn ptr_clone(&self) -> Self {
        Self(self.0)
    }
}
impl StatePtr {
    #[doc(hidden)]
    pub unsafe fn _new(ptr: NonNull<u8>) -> Self {
        Self(ptr)
    }

    pub unsafe fn as_ref<T>(&self) -> &T {
        unsafe { self.0.as_ptr().cast::<T>().as_ref() }.unwrap()
    }

    pub unsafe fn as_mut<T>(&self) -> &mut T {
        unsafe { self.0.as_ptr().cast::<T>().as_mut() }.unwrap()
    }

    pub unsafe fn as_ptr<T>(&self) -> *mut T {
        unsafe { self.0.as_ptr().cast::<T>() }
    }

    unsafe fn ptr_clone(&self) -> Self {
        Self(self.0)
    }
}

pub trait StatefulUdfTrait: Send + Sync + Sized {
    type State: Send + Sync + Sized;

    // Planning methods
    fn flags(&self) -> UdfV2Flags;
    fn format(&self) -> &str;
    fn to_field(&self, fields: &Schema) -> PolarsResult<Field>;

    // Execution methods
    fn initialize(&self, fields: &Schema) -> PolarsResult<Self::State>;
    fn insert(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>>;
    fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>>;
    fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()> {
        _ = (state, other);
        Ok(())
    }
    fn new_empty(&self, state: &Self::State, fields: &Schema) -> PolarsResult<Self::State>;
    fn reset(&self, state: &mut Self::State, fields: &Schema) -> PolarsResult<()>;
}

#[macro_export]
macro_rules! new_udf_vtable {
    ($data:path, $state:path) => {{
        use ::polars_core::prelude::{Field, Schema};
        use ::polars_core::series::Series;
        use ::polars_error::PolarsResult;

        unsafe fn initialize(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            fields: &Schema,
        ) -> PolarsResult<$crate::dsl::expr::anonymous::v2::StatePtr> {
            let data = unsafe { data.as_ref::<$data>() };
            let out = <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::initialize(
                data, fields,
            )?;
            let out = ::std::boxed::Box::new(out);
            let out = ::std::boxed::Box::into_raw(out);
            let out = ::std::ptr::NonNull::new(out as *mut u8).unwrap();
            let out = $crate::dsl::expr::anonymous::v2::StatePtr::_new(out);
            Ok(out)
        }

        unsafe fn insert(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            state: $crate::dsl::expr::anonymous::v2::StatePtr,
            inputs: &[Series],
        ) -> PolarsResult<Option<Series>> {
            let data = unsafe { data.as_ref::<$data>() };
            let state = unsafe { state.as_mut::<$state>() };
            <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::insert(
                data, state, inputs,
            )
        }

        unsafe fn finalize(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            state: $crate::dsl::expr::anonymous::v2::StatePtr,
        ) -> PolarsResult<Option<Series>> {
            let data = unsafe { data.as_ref::<$data>() };
            let state = unsafe { state.as_mut::<$state>() };
            <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::finalize(data, state)
        }

        unsafe fn combine(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            state: $crate::dsl::expr::anonymous::v2::StatePtr,
            other: $crate::dsl::expr::anonymous::v2::StatePtr,
        ) -> PolarsResult<()> {
            let data = unsafe { data.as_ref::<$data>() };
            let state = unsafe { state.as_mut::<$state>() };
            let other = unsafe { other.as_ref::<$state>() };
            <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::combine(
                data, state, other,
            )
        }

        unsafe fn new_empty(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            state: $crate::dsl::expr::anonymous::v2::StatePtr,
            fields: &Schema,
        ) -> PolarsResult<$crate::dsl::expr::anonymous::v2::StatePtr> {
            let data = unsafe { data.as_ref::<$data>() };
            let state = unsafe { state.as_ref::<$state>() };
            let out = <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::new_empty(
                data, state, fields,
            )?;
            let out = ::std::boxed::Box::new(out);
            let out = ::std::boxed::Box::into_raw(out);
            let out = ::std::ptr::NonNull::new(out as *mut u8).unwrap();
            let out = $crate::dsl::expr::anonymous::v2::StatePtr::_new(out);
            Ok(out)
        }

        unsafe fn reset(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            state: $crate::dsl::expr::anonymous::v2::StatePtr,
            fields: &Schema,
        ) -> PolarsResult<()> {
            let data = unsafe { data.as_ref::<$data>() };
            let state = unsafe { state.as_mut::<$state>() };
            <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::reset(
                data, state, fields,
            )
        }

        unsafe fn to_field(
            data: $crate::dsl::expr::anonymous::v2::DataPtr,
            fields: &Schema,
        ) -> PolarsResult<Field> {
            let data = unsafe { data.as_ref::<$data>() };
            <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::to_field(data, fields)
        }

        unsafe fn format(data: $crate::dsl::expr::anonymous::v2::DataPtr) -> (*const u8, usize) {
            let data = unsafe { data.as_ref::<$data>() };
            let format =
                <$data as $crate::dsl::expr::anonymous::v2::StatefulUdfTrait>::format(data);
            (format.as_ptr(), format.len())
        }

        unsafe fn drop_state(state: $crate::dsl::expr::anonymous::v2::StatePtr) {
            let state = unsafe { state.as_ptr::<$data>() };
            let state = unsafe { Box::from_raw(state) };
            drop(state)
        }

        unsafe fn drop_data(data: $crate::dsl::expr::anonymous::v2::DataPtr) {
            let data = unsafe { data.as_ptr::<$data>() };
            let data = unsafe { Box::from_raw(data) };
            drop(data)
        }

        let vtable = unsafe {
            $crate::dsl::expr::anonymous::v2::StatefulUdfVTable::_new(
                initialize, insert, finalize, combine, new_empty, reset, to_field, format,
                drop_state, drop_data,
            )
        };
        $crate::dsl::expr::anonymous::v2::UdfVTable<$data, $state> {
            _pd: ::std::marker::PhantomData,
            vtable: Arc::new(vtable),
        }
    }};
}

bitflags::bitflags! {
    #[derive(Debug, Clone, Copy)]
    pub struct UdfV2Flags: u64 {
        // Function flags.
        /// Preserves length of first non-scalar input.
        ///
        /// If all inputs are scalars, the output is also a scalar.
        const LENGTH_PRESERVING   = 0x01;
        /// Given a function f and a column of values [v1, ..., vn]
        /// f is row-separable i.f.f.
        /// f([v1, ..., vn]) = concat(f(v1, ... vm), f(vm+1, ..., vn))
        /// where scalar input are broadcasted over the length of the other inputs.
        ///
        /// Only makes sense if the inputs are zippable.
        const ROW_SEPARABLE       = 0x02;
        /// Output is always a single row.
        const RETURNS_SCALAR      = 0x04;
        /// All inputs are expected to be of equal length or scalars.
        const ZIPPABLE_INPUTS     = 0x08;

        // Evaluation related flags.
        /// Inserting can yield data.
        const INSERT_HAS_OUTPUT   = 0x10;
        /// Finalize needs to be called and may yield data.
        ///
        /// Expressions that are both LENGTH_PRESERVING and ROW_SEPARABLE are never finalized.
        const NEEDS_FINALIZE      = 0x20;
        /// States can be inserted separately and combined later.
        ///
        /// If a finalization is needed, it is only called one state that has combined all states.
        ///
        /// Expressions that are both LENGTH_PRESERVING and ROW_SEPARABLE are never combined.
        const STATES_COMBINABLE   = 0x40;

        // Expression expansion related flags.
        /// Expand selectors as individual inputs.
        const SELECTOR_EXPANSION  = 0x80;
    }
}

impl UdfV2Flags {
    pub fn is_elementwise(&self) -> bool {
        self.contains(Self::LENGTH_PRESERVING | Self::ROW_SEPARABLE)
    }

    pub fn needs_finalize(&self) -> bool {
        !self.is_elementwise() && self.contains(Self::NEEDS_FINALIZE)
    }

    pub fn allows_concurrent_evaluation(&self) -> bool {
        self.is_elementwise() || self.contains(Self::STATES_COMBINABLE)
    }
}

pub struct StatefulUdf {
    flags: UdfV2Flags,
    formatted: Box<str>,
    data: DataPtr,
    vtable: StatefulUdfVTable,
}

pub struct UdfVTable<Data, State> {
    _pd: PhantomData<(Data, State)>,
    vtable: StatefulUdfVTable,
}

const STUDF_VERSION: u32 = 0x00_00_01;

impl<Data, State> UdfVTable<Data, State> {
    #[cfg(feature = "python")]
    pub fn new_capsule<'py>(
        self,
        py: pyo3::Python<'py>,
        data: Data,
    ) -> pyo3::PyResult<pyo3::Bound<'py, pyo3::types::PyList>> {
        use pyo3::IntoPyObjectExt;

        let data = Box::new(data);
        let data = Box::into_raw(data) as *const u8;
        pyo3::types::PyList::new(
            py,
            [
                STUDF_VERSION.into_py_any(py)?,
                (data as usize).into_py_any(py)?,
                [
                    self.vtable.initialize as usize, // 0
                    self.vtable.new_empty as usize,  // 1
                    self.vtable.reset as usize,      // 2
                    self.vtable.insert as usize,     // 3
                    self.vtable.finalize as usize,   // 4
                    self.vtable.combine as usize,    // 5
                    self.vtable.to_field as usize,   // 6
                    self.vtable.flags as usize,      // 7
                    self.vtable.format as usize,     // 8
                    self.vtable.drop_state as usize, // 9
                    self.vtable.drop_data as usize,  // 10
                ]
                .into_py_any(py)?,
            ],
        )
    }
}

type InitializeFfiFn = unsafe extern "C" fn(
    DataPtr,
    NonNull<ArrowSchema>,
    NonNull<MaybeUninit<StatePtr>>,
) -> ::std::ffi::c_int;
type NewEmptyFfiFn = unsafe extern "C" fn(
    DataPtr,
    StatePtr,
    NonNull<ArrowSchema>,
    StatePtr,
    NonNull<MaybeUninit<StatePtr>>,
) -> ::std::ffi::c_int;
type ResetFfiFn =
    unsafe extern "C" fn(DataPtr, StatePtr, NonNull<ArrowSchema>) -> ::std::ffi::c_int;
type InsertFfiFn = unsafe extern "C" fn(
    DataPtr,
    StatePtr,
    NonNull<SeriesExport>,
    usize,
    NonNull<MaybeUninit<SeriesExport>>,
) -> ::std::ffi::c_int;
type FinalizeFfiFn = unsafe extern "C" fn(
    DataPtr,
    StatePtr,
    NonNull<MaybeUninit<SeriesExport>>,
) -> ::std::ffi::c_int;
type CombineFfiFn = unsafe extern "C" fn(DataPtr, StatePtr, StatePtr) -> ::std::ffi::c_int;
type ToFieldFfiFn = unsafe extern "C" fn(
    DataPtr,
    NonNull<ArrowSchema>,
    NonNull<MaybeUninit<ArrowSchema>>,
) -> ::std::ffi::c_int;

#[derive(Clone)]
pub struct StatefulUdfVTable {
    initialize: InitializeFfiFn,
    new_empty: NewEmptyFfiFn,
    reset: ResetFfiFn,

    insert: InsertFfiFn,
    finalize: FinalizeFfiFn,
    combine: CombineFfiFn,

    to_field: ToFieldFfiFn,
    flags: unsafe extern "C" fn(DataPtr) -> UdfV2Flags,
    format: unsafe extern "C" fn(DataPtr) -> (NonNull<u8>, usize),
    drop_state: unsafe extern "C" fn(StatePtr),
    drop_data: unsafe extern "C" fn(DataPtr),
}

unsafe impl Sync for StatefulUdf {}
unsafe impl Send for StatefulUdf {}

unsafe impl Send for UdfState {}
unsafe impl Sync for UdfState {}

#[cfg(feature = "serde")]
impl serde::Serialize for StatefulUdf {
    fn serialize<S>(&self, serializer: S) -> Result<S::Ok, S::Error>
    where
        S: serde::Serializer,
    {
        use serde::ser::Error;
        Err(S::Error::custom(
            "serialization not supported for this 'opaque' function",
        ))
    }
}

#[cfg(feature = "serde")]
impl<'de> serde::Deserialize<'de> for StatefulUdf {
    fn deserialize<D>(deserializer: D) -> Result<Self, D::Error>
    where
        D: serde::Deserializer<'de>,
    {
        use serde::de::Error;
        Err(D::Error::custom(
            "deserialization not supported for this 'opaque' function",
        ))
    }
}

#[cfg(feature = "dsl-schema")]
impl schemars::JsonSchema for StatefulUdf {
    fn schema_name() -> String {
        "StatefulUdf".to_owned()
    }

    fn schema_id() -> std::borrow::Cow<'static, str> {
        std::borrow::Cow::Borrowed(concat!(module_path!(), "::", "StatefulUdf"))
    }

    fn json_schema(generator: &mut schemars::r#gen::SchemaGenerator) -> schemars::schema::Schema {
        Vec::<u8>::json_schema(generator)
    }
}

impl StatefulUdf {
    pub fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
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
        let rt = unsafe {
            (self.vtable.to_field)(
                self.data.ptr_clone(),
                NonNull::from_ref(&fields),
                NonNull::from_mut(&mut field),
            )
        };

        if rt != 0 {
            // @TODO: Improve error message.
            polars_bail!(ComputeError: "error occured in plugin 'to_field'");
        }

        let field = unsafe { field.assume_init() };
        let field = unsafe { arrow::ffi::import_field_from_c(&field) }?;
        Ok(Field::from(&field))
    }

    pub fn initialize(self: Arc<Self>, fields: &Schema) -> PolarsResult<UdfState> {
        let mut state: MaybeUninit<StatePtr> = MaybeUninit::uninit();
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
        let rt = unsafe {
            (self.vtable.initialize)(
                self.data.ptr_clone(),
                NonNull::from_ref(&fields),
                NonNull::from_mut(&mut state),
            )
        };

        if rt != 0 {
            // @TODO: Improve error message.
            polars_bail!(ComputeError: "error occured in plugin 'initialize'");
        }

        let ptr = unsafe { state.assume_init() };
        Ok(UdfState { ptr, udf: self })
    }

    pub fn flags(&self) -> UdfV2Flags {
        self.flags
    }

    pub fn format_string(&self) -> &str {
        &self.formatted
    }

    #[doc(hidden)]
    pub unsafe fn _new(
        flags: UdfV2Flags,
        formatted: Box<str>,
        data: DataPtr,
        vtable: StatefulUdfVTable,
    ) -> Self {
        Self {
            flags,
            formatted,
            data,
            vtable,
        }
    }
}

impl StatefulUdfVTable {
    #[doc(hidden)]
    pub unsafe fn _new(
        initialize: InitializeFfiFn,
        insert: InsertFfiFn,
        finalize: FinalizeFfiFn,
        combine: CombineFfiFn,
        new_empty: NewEmptyFfiFn,
        reset: ResetFfiFn,

        to_field: unsafe fn(DataPtr, &Schema) -> PolarsResult<Field>,
        drop_state: unsafe fn(StatePtr),
        drop_data: unsafe fn(DataPtr),
    ) -> Self {
        Self {
            initialize,
            insert,
            finalize,
            combine,
            new_empty,
            reset,
            to_field,
            drop_state,
            drop_data,
        }
    }
}

impl Drop for StatefulUdf {
    fn drop(&mut self) {
        unsafe { (self.vtable.drop_data)(self.data.ptr_clone()) }
    }
}

impl Drop for UdfState {
    fn drop(&mut self) {
        unsafe { (self.udf.vtable.drop_state)(self.ptr.ptr_clone()) }
    }
}

pub struct UdfState {
    ptr: StatePtr,
    udf: Arc<StatefulUdf>,
}

impl UdfState {
    pub fn insert(&mut self, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        unsafe { (self.udf.vtable.insert)(self.udf.data.ptr_clone(), self.ptr.ptr_clone(), inputs) }
    }

    pub fn finalize(&mut self) -> PolarsResult<Option<Series>> {
        assert!(self.udf.flags.contains(UdfV2Flags::NEEDS_FINALIZE));
        unsafe { (self.udf.vtable.finalize)(self.udf.data.ptr_clone(), self.ptr.ptr_clone()) }
    }

    pub fn combine(&mut self, other: &Self) -> PolarsResult<()> {
        assert_eq!(Arc::as_ptr(&self.udf), Arc::as_ptr(&other.udf));
        assert!(self.udf.flags.contains(UdfV2Flags::STATES_COMBINABLE));
        unsafe {
            (self.udf.vtable.combine)(
                self.udf.data.ptr_clone(),
                self.ptr.ptr_clone(),
                other.ptr.ptr_clone(),
            )
        }
    }

    pub fn new_empty(&self, schema: &Schema) -> PolarsResult<UdfState> {
        let udf = self.udf.clone();
        let ptr = unsafe {
            (self.udf.vtable.new_empty)(udf.data.ptr_clone(), self.ptr.ptr_clone(), schema)
        }?;
        Ok(UdfState { ptr, udf })
    }

    pub fn reset(&mut self, schema: &Schema) -> PolarsResult<()> {
        unsafe { (self.udf.vtable.reset)(self.udf.data.ptr_clone(), self.ptr.ptr_clone(), schema) }
    }
}
