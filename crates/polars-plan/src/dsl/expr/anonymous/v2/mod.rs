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
use polars_utils::pl_str::PlSmallStr;

pub mod ffi;
#[cfg(feature = "serde")]
mod serde;

#[macro_export]
macro_rules! polars_plugin_expr_info {
    (
        $name:literal, $data:expr, $data_ty:ty
    ) => {{
        #[unsafe(export_name = concat!("_PL_PLUGIN_V2::", $name))]
        pub static VTABLE: $crate::prelude::v2::PresentedUdf = $crate::prelude::v2::PresentedUdf {
            version: $crate::prelude::v2::STUDF_VERSION,
            vtable: $crate::prelude::v2::ffi::VTable::new::<$data_ty>(),
        };

        let data = ::std::boxed::Box::new($data);
        let data = ::std::boxed::Box::into_raw(data);
        $crate::prelude::v2::PolarsPluginExprInfo::_new($name, data as *const u8)
    }};
}

#[repr(transparent)]
pub struct DataPtr(NonNull<u8>);
#[repr(transparent)]
pub struct StatePtr(NonNull<u8>);

impl DataPtr {
    #[doc(hidden)]
    pub fn _new(ptr: NonNull<u8>) -> Self {
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

    // (De)serialization methods.
    fn serialize(&self) -> PolarsResult<Box<[u8]>>;
    fn deserialize(data: &[u8]) -> PolarsResult<Self>;
    fn serialize_state(&self, state: &Self::State) -> PolarsResult<Box<[u8]>>;
    fn deserialize_state(&self, data: &[u8]) -> PolarsResult<Self::State>;

    // Planning methods.
    fn to_field(&self, fields: &Schema) -> PolarsResult<Field>;

    // State management methods.
    fn initialize(&self, fields: &Schema) -> PolarsResult<Self::State>;
    fn new_empty(&self, state: &Self::State) -> PolarsResult<Self::State>;
    fn reset(&self, state: &mut Self::State) -> PolarsResult<()>;
    fn combine(&self, state: &mut Self::State, other: &Self::State) -> PolarsResult<()> {
        _ = (state, other);
        Ok(())
    }

    // Execution methods.
    fn insert(&self, state: &mut Self::State, inputs: &[Series]) -> PolarsResult<Option<Series>>;
    fn finalize(&self, state: &mut Self::State) -> PolarsResult<Option<Series>>;
}

pub const fn new_udf_vtable<Data: StatefulUdfTrait>() -> UdfVTable<Data> {
    UdfVTable {
        _pd: ::std::marker::PhantomData,
        vtable: ffi::VTable::new::<Data>(),
    }
}

pub struct PolarsPluginExprInfo {
    symbol: &'static str,
    data_ptr: *const u8,
}

impl PolarsPluginExprInfo {
    #[doc(hidden)]
    pub fn _new(symbol: &'static str, data_ptr: *const u8) -> Self {
        Self { symbol, data_ptr }
    }
}

#[cfg(feature = "python")]
impl<'py> pyo3::IntoPyObject<'py> for PolarsPluginExprInfo {
    type Target = pyo3::types::PyTuple;
    type Output = pyo3::Bound<'py, Self::Target>;
    type Error = pyo3::PyErr;

    fn into_pyobject(self, py: pyo3::Python<'py>) -> Result<Self::Output, Self::Error> {
        use pyo3::{IntoPyObject, IntoPyObjectExt};
        pyo3::types::PyTuple::new(
            py,
            [
                self.symbol.into_py_any(py)?,
                (self.data_ptr as usize).into_py_any(py)?,
            ],
        )
    }
}

bitflags::bitflags! {
    #[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
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

    pub fn verify_coherency(&self) -> PolarsResult<()> {
        if self.contains(Self::LENGTH_PRESERVING | Self::RETURNS_SCALAR) {
            polars_bail!(InvalidOperation: "expression cannot be both `length_preserving` and `returns_scalar`");
        }

        Ok(())
    }
}

struct LibrarySymbol {
    lib: PlSmallStr,
    symbol: PlSmallStr,
    library: libloading::Library,
}

pub struct StatefulUdf {
    flags: UdfV2Flags,
    function_name: PlSmallStr,
    data: DataPtr,
    library: Option<Box<LibrarySymbol>>,
    vtable: ffi::VTable,
}

pub struct UdfVTable<Data: StatefulUdfTrait> {
    _pd: PhantomData<Data>,
    vtable: ffi::VTable,
}

#[doc(hidden)]
pub const STUDF_VERSION: u32 = 0x00_00_01;

#[doc(hidden)]
#[repr(C)]
pub struct PresentedUdf {
    pub version: u32,
    pub vtable: ffi::VTable,
}

impl<Data: StatefulUdfTrait> UdfVTable<Data> {
    pub unsafe fn new_udf(
        self,
        data: DataPtr,
        flags: UdfV2Flags,
        function_name: PlSmallStr,
    ) -> StatefulUdf {
        StatefulUdf {
            flags,
            function_name,
            data,
            library: None,
            vtable: self.vtable,
        }
    }
}

unsafe impl Sync for StatefulUdf {}
unsafe impl Send for StatefulUdf {}

unsafe impl Send for UdfState {}
unsafe impl Sync for UdfState {}

fn load_vtable(lib: &str, symbol: &str) -> PolarsResult<(libloading::Library, ffi::VTable)> {
    let lib = unsafe { libloading::Library::new(lib) }.unwrap();
    let name = format!("_PL_PLUGIN_V2::{symbol}");
    let udf: libloading::Symbol<NonNull<PresentedUdf>> =
        unsafe { lib.get(name.as_bytes()) }.unwrap();
    let vtable = unsafe { udf.as_ref() }.vtable.clone();
    Ok((lib, vtable))
}

impl StatefulUdf {
    pub unsafe fn new_shared_object(
        library: &str,
        symbol: &str,
        data: usize,
        flags: UdfV2Flags,
        function_name: PlSmallStr,
    ) -> PolarsResult<StatefulUdf> {
        flags.verify_coherency()?;

        let data = DataPtr(NonNull::new(data as *mut u8).unwrap());
        let (lib, vtable) = load_vtable(&library, &symbol)?;

        Ok(StatefulUdf {
            flags,
            function_name,
            data,
            library: Some(Box::new(LibrarySymbol {
                lib: library.into(),
                symbol: symbol.into(),
                library: lib,
            })),
            vtable,
        })
    }

    pub fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        unsafe { self.vtable.to_field(self.data.ptr_clone(), fields) }
    }

    pub fn initialize(self: Arc<Self>, fields: &Schema) -> PolarsResult<UdfState> {
        let ptr = unsafe { self.vtable.initialize(self.data.ptr_clone(), fields) }?;
        Ok(UdfState { ptr, udf: self })
    }

    pub fn flags(&self) -> UdfV2Flags {
        self.flags
    }

    pub fn name(&self) -> &str {
        &self.function_name
    }
}

impl Drop for StatefulUdf {
    fn drop(&mut self) {
        unsafe { self.vtable.drop_data(self.data.ptr_clone()) }
    }
}

impl Drop for UdfState {
    fn drop(&mut self) {
        unsafe { self.udf.vtable.drop_state(self.ptr.ptr_clone()) }
    }
}

pub struct UdfState {
    ptr: StatePtr,
    udf: Arc<StatefulUdf>,
}

impl UdfState {
    pub fn insert(&mut self, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        unsafe {
            self.udf
                .vtable
                .insert(self.udf.data.ptr_clone(), self.ptr.ptr_clone(), inputs)
        }
    }

    pub fn finalize(&mut self) -> PolarsResult<Option<Series>> {
        assert!(self.udf.flags.contains(UdfV2Flags::NEEDS_FINALIZE));
        unsafe {
            self.udf
                .vtable
                .finalize(self.udf.data.ptr_clone(), self.ptr.ptr_clone())
        }
    }

    pub fn combine(&mut self, other: &Self) -> PolarsResult<()> {
        assert_eq!(Arc::as_ptr(&self.udf), Arc::as_ptr(&other.udf));
        assert!(self.udf.flags.contains(UdfV2Flags::STATES_COMBINABLE));
        unsafe {
            self.udf.vtable.combine(
                self.udf.data.ptr_clone(),
                self.ptr.ptr_clone(),
                other.ptr.ptr_clone(),
            )
        }
    }

    pub fn new_empty(&self) -> PolarsResult<UdfState> {
        let udf = self.udf.clone();
        let ptr = unsafe {
            self.udf
                .vtable
                .new_empty(udf.data.ptr_clone(), self.ptr.ptr_clone())
        }?;
        Ok(UdfState { ptr, udf })
    }

    pub fn reset(&mut self) -> PolarsResult<()> {
        unsafe {
            self.udf
                .vtable
                .reset(self.udf.data.ptr_clone(), self.ptr.ptr_clone())
        }
    }
}
