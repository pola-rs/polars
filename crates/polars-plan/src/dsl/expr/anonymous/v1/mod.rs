use core::str;
use std::alloc::Layout;
use std::marker::PhantomData;
use std::mem::MaybeUninit;
use std::ptr::NonNull;
use std::sync::Arc;

use arrow::array::Array;
use arrow::bitmap::Bitmap;
use arrow::datatypes::{ArrowDataType, Field as ArrowField};
use arrow::ffi::ArrowSchema;
use polars_core::frame::DataFrame;
use polars_core::prelude::{CompatLevel, Field};
use polars_core::schema::{Schema, SchemaExt};
use polars_core::series::Series;
use polars_error::{PolarsResult, polars_bail};
use polars_ffi::version_1 as ffi;
use polars_utils::pl_str::PlSmallStr;

#[cfg(feature = "serde")]
mod serde;

pub struct PluginV1 {
    flags: PluginV1Flags,
    function_name: PlSmallStr,
    data: ffi::DataPtr,
    library: Option<Box<LibrarySymbol>>,
    vtable: ffi::VTable,
}

pub struct PluginV1State {
    ptr: ffi::StatePtr,
    plugin: Arc<PluginV1>,
}

struct LibrarySymbol {
    lib: PlSmallStr,
    symbol: PlSmallStr,
    library: libloading::Library,
}

bitflags::bitflags! {
    #[cfg_attr(feature = "serde", derive(::serde::Serialize, ::serde::Deserialize))]
    #[derive(Debug, Clone, Copy)]
    pub struct PluginV1Flags: u64 {
        // Function flags.
        /// Preserves length of first non-scalar input.
        ///
        /// If all inputs are scalars, the output is also a scalar.
        const LENGTH_PRESERVING   = 0x001;
        /// Given a function f and a column of values [v1, ..., vn]
        /// f is row-separable i.f.f.
        /// f([v1, ..., vn]) = concat(f(v1, ... vm), f(vm+1, ..., vn))
        /// where scalar input are broadcasted over the length of the other inputs.
        ///
        /// Only makes sense if the inputs are zippable.
        const ROW_SEPARABLE       = 0x002;
        /// Output is always a single row.
        const RETURNS_SCALAR      = 0x004;
        /// All inputs are expected to be of equal length or scalars.
        const ZIPPABLE_INPUTS     = 0x008;

        // Evaluation related flags.
        /// Each step may yield data.
        const STEP_HAS_OUTPUT   = 0x010;
        /// Finalize needs to be called and may yield data.
        ///
        /// Expressions that are both LENGTH_PRESERVING and ROW_SEPARABLE are never finalized.
        const NEEDS_FINALIZE      = 0x020;
        /// States can be created separately and combined later.
        ///
        /// If a finalization is needed, it is only called one state that has combined all states.
        ///
        /// Expressions that are both LENGTH_PRESERVING and ROW_SEPARABLE are never combined.
        const STATES_COMBINABLE   = 0x040;
        /// Group evaluation is specialized.
        const SPECIALIZE_GROUP_EVALUATION   = 0x080;

        // Expression expansion related flags.
        /// Expand selectors as individual inputs.
        const SELECTOR_EXPANSION  = 0x100;
    }
}

impl PluginV1Flags {
    pub fn is_elementwise(&self) -> bool {
        self.contains(Self::LENGTH_PRESERVING | Self::ROW_SEPARABLE)
    }

    pub fn is_length_preserving(&self) -> bool {
        self.contains(Self::LENGTH_PRESERVING)
    }

    pub fn returns_scalar(&self) -> bool {
        self.contains(Self::RETURNS_SCALAR)
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

unsafe impl Sync for PluginV1 {}
unsafe impl Send for PluginV1 {}

unsafe impl Send for PluginV1State {}
unsafe impl Sync for PluginV1State {}

fn load_vtable(lib: &str, symbol: &str) -> PolarsResult<(libloading::Library, ffi::VTable)> {
    let lib = unsafe { libloading::Library::new(lib) }.unwrap();
    let name = format!("_PL_PLUGIN_V2::{symbol}");
    let plugin: libloading::Symbol<NonNull<ffi::PluginSymbol>> =
        unsafe { lib.get(name.as_bytes()) }.unwrap();
    let vtable = unsafe { plugin.as_ref() }.vtable.clone();
    unsafe { vtable.set_version(ffi::VERSION) };
    Ok((lib, vtable))
}

impl PluginV1 {
    /// # Safety
    ///
    /// This is inherently unsafe, we are loading a shared library from an arbitrary location.
    /// Trust the shared library??
    pub unsafe fn new_shared_object(
        lib: &str,
        symbol: &str,
        data: usize,
        flags: PluginV1Flags,
        function_name: PlSmallStr,
    ) -> PolarsResult<PluginV1> {
        flags.verify_coherency()?;

        let data = ffi::DataPtr::_new(NonNull::new(data as *mut u8).unwrap());
        let (library, vtable) = load_vtable(lib, symbol)?;

        Ok(PluginV1 {
            flags,
            function_name,
            data,
            library: Some(Box::new(LibrarySymbol {
                lib: lib.into(),
                symbol: symbol.into(),
                library,
            })),
            vtable,
        })
    }

    pub fn to_field(&self, fields: &Schema) -> PolarsResult<Field> {
        unsafe { self.vtable.to_field(self.data.ptr_clone(), fields) }
    }

    pub fn initialize(self: Arc<Self>, fields: &Schema) -> PolarsResult<PluginV1State> {
        let ptr = unsafe { self.vtable.new_state(self.data.ptr_clone(), fields) }?;
        Ok(PluginV1State { ptr, plugin: self })
    }

    pub fn evaluate_on_groups<'a>(
        &self,
        inputs: &[(Series, &'a ffi::GroupPositions)],
    ) -> PolarsResult<(Series, ffi::CowGroupPositions<'a>)> {
        unsafe {
            self.vtable
                .evaluate_on_groups(self.data.ptr_clone(), inputs)
        }
    }

    pub fn flags(&self) -> PluginV1Flags {
        self.flags
    }

    pub fn function_name(&self) -> &str {
        &self.function_name
    }
}

impl Drop for PluginV1 {
    fn drop(&mut self) {
        unsafe { self.vtable.drop_data(self.data.ptr_clone()) }
    }
}

impl Drop for PluginV1State {
    fn drop(&mut self) {
        unsafe { self.plugin.vtable.drop_state(self.ptr.ptr_clone()) }
    }
}

impl PluginV1State {
    pub fn step(&mut self, inputs: &[Series]) -> PolarsResult<Option<Series>> {
        unsafe {
            self.plugin
                .vtable
                .step(self.plugin.data.ptr_clone(), self.ptr.ptr_clone(), inputs)
        }
    }

    pub fn finalize(&mut self) -> PolarsResult<Option<Series>> {
        assert!(self.plugin.flags.contains(PluginV1Flags::NEEDS_FINALIZE));
        unsafe {
            self.plugin
                .vtable
                .finalize(self.plugin.data.ptr_clone(), self.ptr.ptr_clone())
        }
    }

    pub fn combine(&mut self, other: &Self) -> PolarsResult<()> {
        assert_eq!(Arc::as_ptr(&self.plugin), Arc::as_ptr(&other.plugin));
        assert!(self.plugin.flags.contains(PluginV1Flags::STATES_COMBINABLE));
        unsafe {
            self.plugin.vtable.combine(
                self.plugin.data.ptr_clone(),
                self.ptr.ptr_clone(),
                other.ptr.ptr_clone(),
            )
        }
    }

    pub fn new_empty(&self) -> PolarsResult<PluginV1State> {
        let plugin = self.plugin.clone();
        let ptr = unsafe {
            self.plugin
                .vtable
                .new_empty(plugin.data.ptr_clone(), self.ptr.ptr_clone())
        }?;
        Ok(PluginV1State { ptr, plugin })
    }

    pub fn reset(&mut self) -> PolarsResult<()> {
        unsafe {
            self.plugin
                .vtable
                .reset(self.plugin.data.ptr_clone(), self.ptr.ptr_clone())
        }
    }

    pub fn serialize(&self, out: &mut Vec<u8>) -> PolarsResult<()> {
        unsafe {
            self.plugin.vtable.serialize_state(
                self.plugin.data.ptr_clone(),
                self.ptr.ptr_clone(),
                out,
            )
        }
    }

    pub fn deserialize(plugin: Arc<PluginV1>, buffer: &[u8]) -> PolarsResult<Self> {
        let ptr = unsafe {
            plugin
                .vtable
                .deserialize_state(plugin.data.ptr_clone(), buffer)
        }?;
        Ok(PluginV1State { ptr, plugin })
    }
}
