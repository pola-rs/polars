use std::any::Any;
use std::borrow::Cow;
use std::fmt::{Display, Debug};
use std::hash::{Hash, Hasher};

use crate::datatypes::DataType;

mod registry;
mod generic;

use generic::GenericExtensionType;
pub use registry::{get_extension_type_or_generic, get_extension_type_or_storage, register_extension_type, unregister_extension_type};


pub trait ExtensionTypeFactory: 'static + Send + Sync {
    fn create_type_instance(&self, name: &str, storage: &DataType, metadata: Option<&str>) -> Box<dyn ExtensionTypeImpl>;
}

pub trait ExtensionTypeImpl : 'static + Send + Sync + Any {
    /// Name of the extension type.
    fn name(&self) -> Cow<'_, str>;
    
    /// Serialize the metadata of the extension type.
    fn serialize_metadata(&self) -> Option<Cow<'_, str>>;

    fn dyn_clone(&self) -> Box<dyn ExtensionTypeImpl>;
    fn dyn_eq(&self, other: &dyn ExtensionTypeImpl) -> bool;
    fn dyn_hash(&self) -> u64;

    /// Display representation of the extension type.
    /// 
    /// Should be a short string representation, lowercase. For example: str, datetime[ms].
    fn dyn_display(&self) -> Cow<'_, str>;
    
    /// Debug representation of the extension type.
    /// 
    /// Should be a more verbose string representation, useful for debugging, in TitleCase,
    /// for example: String, Decimal(10, 2).
    fn dyn_debug(&self) -> Cow<'_, str>;
}

#[repr(transparent)]
pub struct ExtensionTypeInstance(pub Box<dyn ExtensionTypeImpl>);

impl Clone for ExtensionTypeInstance {
    fn clone(&self) -> Self {
        Self(self.0.dyn_clone())
    }
}

impl PartialEq for ExtensionTypeInstance {
    fn eq(&self, other: &Self) -> bool {
        self.0.dyn_eq(&*other.0)
    }
}

impl Eq for ExtensionTypeInstance {}

impl Hash for ExtensionTypeInstance {
    fn hash<H: Hasher>(&self, state: &mut H) {
        let h = self.0.dyn_hash();
        h.hash(state);
    }
}

impl Display for ExtensionTypeInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.dyn_display())
    }
}

impl Debug for ExtensionTypeInstance {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(f, "{}", self.0.dyn_debug())
    }
}

impl ExtensionTypeInstance {
    pub fn name(&self) -> Cow<'_, str> {
        self.0.name()
    }

    pub fn serialize_metadata(&self) -> Option<Cow<'_, str>> {
        self.0.serialize_metadata()
    }
}




// pub mod vtable;

/*

pub use vtable::ExtensionParams;
use vtable::{OutBuffer, VTable};

// We use a separate lock for registering so we can hold the lock for longer
// periods during e.g. plugin registration without blocking reading.
static REGISTRY_UPDATE_LOCK: Mutex<()> = Mutex::new(());

static REGISTRY: LazyLock<RwLock<HashMap<PlSmallStr, Arc<ExtensionType>>>> =
    LazyLock::new(|| RwLock::default());

pub fn get_extension_type(name: &str) -> PolarsResult<Arc<ExtensionType>> {
    REGISTRY.read().unwrap().get(name).cloned().ok_or_else(|| {
        // TODO @ extension-type: a better error type?
        polars_err!(ComputeError: "unknown extension type '{name}'")
    })
}

pub fn register_extension_type(t: ExtensionType) -> PolarsResult<()> {
    let _lock = REGISTRY_UPDATE_LOCK.lock().unwrap();

    match REGISTRY.write().unwrap().entry(t.name.clone()) {
        Entry::Occupied(_) => {
            polars_bail!(ComputeError: "attempted to register duplicate extension type with name '{}'", t.name)
        },
        Entry::Vacant(v) => {
            v.insert(Arc::new(t));
            Ok(())
        },
    }
}

pub struct ExtensionType {
    name: PlSmallStr,
    capabilities: u64,
    vtable: VTable,
}

impl ExtensionType {
    pub fn new(name: PlSmallStr, capabilities: u64, vtable: VTable) -> Self {
        Self {
            name,
            capabilities,
            vtable,
        }
    }

    pub fn name(&self) -> &PlSmallStr {
        &self.name
    }

    pub fn capabilities(&self) -> u64 {
        self.capabilities
    }

    pub fn vtable(&self) -> &VTable {
        &self.vtable
    }

    /// Type is a newtype around a physical representation.
    pub fn is_newtype(&self) -> bool {
        (self.capabilities & EXT_TYPE_CAPABILITY_NEWTYPE) != 0
    }

    /// Type uses the physical representation for hashing and equality checks.
    pub fn phys_hash_eq(&self) -> bool {
        (self.capabilities & EXT_TYPE_CAPABILITY_PHYS_HASH_EQ) != 0
    }

    /// Type uses the physical representation for ordering.
    pub fn phys_ord(&self) -> bool {
        (self.capabilities & EXT_TYPE_CAPABILITY_PHYS_ORD) != 0
    }

    pub fn deserialize_ext_params(&self, buf: &[u8]) -> PolarsResult<ExtensionParams> {
        let mut out = ExtensionParams(std::ptr::null_mut());
        call_ext_method(self, |vt| unsafe {
            (vt.type_deserialize)(buf.as_ptr(), buf.len(), &mut out)
        })?;
        Ok(out)
    }

    pub fn serialize_ext_params(&self, params: ExtensionParams) -> PolarsResult<Vec<u8>> {
        let mut out = OutBuffer::new(Vec::new());
        call_ext_method(self, |vt| unsafe { (vt.type_serialize)(params, &mut out) })?;
        Ok(out.into_inner())
    }

    pub fn display_with(&self, params: ExtensionParams) -> PolarsResult<String> {
        let mut out = OutBuffer::new(Vec::new());
        call_ext_method(self, |vt| unsafe { (vt.type_display)(params, &mut out) })?;
        Ok(String::from_utf8(out.into_inner()).unwrap())
    }

    pub fn debug_with(&self, params: ExtensionParams) -> PolarsResult<String> {
        let mut out = OutBuffer::new(Vec::new());
        call_ext_method(self, |vt| unsafe { (vt.type_debug)(params, &mut out) })?;
        Ok(String::from_utf8(out.into_inner()).unwrap())
    }
}

// Capability flags.
pub const EXT_TYPE_CAPABILITY_NEWTYPE: u64 =
    (1 << 0) | EXT_TYPE_CAPABILITY_PHYS_HASH_EQ | EXT_TYPE_CAPABILITY_PHYS_ORD;
pub const EXT_TYPE_CAPABILITY_PHYS_HASH_EQ: u64 = 1 << 1;
pub const EXT_TYPE_CAPABILITY_PHYS_ORD: u64 = 1 << 2;

fn call_ext_method<F: FnOnce(&VTable) -> bool>(ext: &ExtensionType, f: F) -> PolarsResult<()> {
    let vtable = ext.vtable();
    if f(vtable) {
        let s = unsafe { CStr::from_ptr((vtable.get_last_error)()) };
        polars_bail!(ComputeError: "an extension type failed with message: {}", s.to_string_lossy())
    }
    Ok(())
}

pub struct ExtensionTypeInstance {
    typ: Arc<ExtensionType>,
    params: ExtensionParams,
}

impl Clone for ExtensionTypeInstance {
    fn clone(&self) -> Self {
        let mut out = ExtensionParams(std::ptr::null_mut());
        call_ext_method(&self.typ, |vt| unsafe {
            (vt.type_clone)(self.params, &mut out)
        })
        .unwrap();
        Self {
            typ: self.typ.clone(),
            params: out,
        }
    }
}

impl Drop for ExtensionTypeInstance {
    fn drop(&mut self) {
        call_ext_method(&self.typ, |vt| unsafe {
            (vt.type_drop)(self.params)
        })
        .unwrap()
    }
}

impl Hash for ExtensionTypeInstance {
    fn hash<H: std::hash::Hasher>(&self, state: &mut H) {
        let mut out = 0;
        call_ext_method(&self.typ, |vt| unsafe { (vt.type_hash)(self.params, &mut out) }).unwrap();
        out.hash(state);
    }
}

impl Eq for ExtensionTypeInstance {}

impl PartialEq for ExtensionTypeInstance {
    fn eq(&self, other: &Self) -> bool {
        let mut out = false;
        call_ext_method(&self.typ, |vt| unsafe {
            (vt.type_eq)(self.params, other.params, &mut out)
        })
        .unwrap();
        out
    }
}

impl ExtensionTypeInstance {
    pub fn serialize_params(&self) -> Vec<u8> {
        self.params
    }
}
*/



// fn foo(x: &dyn ExtensionTypeInstance) {
//     x.dyn_clone()
// }