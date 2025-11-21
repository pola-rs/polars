//! This is a heap allocated utility that can be used to register an object type.
//!
//! That object type will know its own generic type parameter `T` and callers can simply
//! send `&Any` values and don't have to know the generic type themselves.
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, LazyLock, RwLock};

use arrow::array::ArrayRef;
use arrow::array::builder::ArrayBuilder;
use arrow::datatypes::ArrowDataType;
use polars_utils::pl_str::PlSmallStr;

use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::datatypes::AnyValue;
use crate::prelude::{ListBuilderTrait, ObjectChunked, PolarsObject};
use crate::series::{IntoSeries, Series};

/// Takes a `name` and `capacity` and constructs a new builder.
pub type BuilderConstructor =
    Box<dyn Fn(PlSmallStr, usize) -> Box<dyn AnonymousObjectBuilder> + Send + Sync>;
pub type ObjectConverter = Arc<dyn Fn(AnyValue) -> Box<dyn Any> + Send + Sync>;
pub type PyObjectConverter = Arc<dyn Fn(AnyValue) -> Box<dyn Any> + Send + Sync>;

pub struct ObjectRegistry {
    /// A function that creates an object builder
    pub builder_constructor: BuilderConstructor,
    // A function that converts AnyValue to Box<dyn Any> of the object type
    object_converter: Option<ObjectConverter>,
    // A function that converts AnyValue to Box<dyn Any> of the PyObject type
    pyobject_converter: Option<PyObjectConverter>,
    pub physical_dtype: ArrowDataType,
}

impl Debug for ObjectRegistry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "object-registry")
    }
}

static GLOBAL_OBJECT_REGISTRY: LazyLock<RwLock<Option<ObjectRegistry>>> =
    LazyLock::new(Default::default);

/// This trait can be registered, after which that global registration
/// can be used to materialize object types
pub trait AnonymousObjectBuilder: ArrayBuilder {
    fn as_array_builder(self: Box<Self>) -> Box<dyn ArrayBuilder>;

    /// # Safety
    /// Expect `ObjectArray<T>` arrays.
    unsafe fn from_chunks(self: Box<Self>, chunks: Vec<ArrayRef>) -> Series;

    /// Append a `null` value.
    fn append_null(&mut self);

    /// Append a `T` of [`ObjectChunked<T>`][ObjectChunked<T>] made generic via the [`Any`] trait.
    ///
    /// [ObjectChunked<T>]: crate::chunked_array::object::ObjectChunked
    fn append_value(&mut self, value: &dyn Any);

    fn append_option(&mut self, value: Option<&dyn Any>) {
        match value {
            None => self.append_null(),
            Some(v) => self.append_value(v),
        }
    }

    /// Take the current state and materialize as a [`Series`]
    /// the builder should not be used after that.
    fn to_series(&mut self) -> Series;

    fn get_list_builder(
        &self,
        name: PlSmallStr,
        values_capacity: usize,
        list_capacity: usize,
    ) -> Box<dyn ListBuilderTrait>;
}

impl<T: PolarsObject> AnonymousObjectBuilder for ObjectChunkedBuilder<T> {
    /// # Safety
    /// Expects `ObjectArray<T>` arrays.
    unsafe fn from_chunks(self: Box<Self>, chunks: Vec<ArrayRef>) -> Series {
        ObjectChunked::<T>::new_with_compute_len(Arc::new(self.field().clone()), chunks)
            .into_series()
    }

    fn as_array_builder(self: Box<Self>) -> Box<dyn ArrayBuilder> {
        self
    }

    fn append_null(&mut self) {
        self.append_null()
    }

    fn append_value(&mut self, value: &dyn Any) {
        let value = value.downcast_ref::<T>().unwrap();
        self.append_value(value.clone())
    }

    fn to_series(&mut self) -> Series {
        let builder = std::mem::take(self);
        builder.finish().into_series()
    }
    fn get_list_builder(
        &self,
        name: PlSmallStr,
        values_capacity: usize,
        list_capacity: usize,
    ) -> Box<dyn ListBuilderTrait> {
        Box::new(super::extension::list::ExtensionListBuilder::<T>::new(
            name,
            values_capacity,
            list_capacity,
        ))
    }
}

pub fn register_object_builder(
    builder_constructor: BuilderConstructor,
    object_converter: ObjectConverter,
    pyobject_converter: PyObjectConverter,
    physical_dtype: ArrowDataType,
) {
    let reg = GLOBAL_OBJECT_REGISTRY.deref();
    let mut reg = reg.write().unwrap();

    *reg = Some(ObjectRegistry {
        builder_constructor,
        object_converter: Some(object_converter),
        pyobject_converter: Some(pyobject_converter),
        physical_dtype,
    })
}

#[cold]
pub fn get_object_physical_type() -> ArrowDataType {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    reg.physical_dtype.clone()
}

pub fn get_object_builder(name: PlSmallStr, capacity: usize) -> Box<dyn AnonymousObjectBuilder> {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    (reg.builder_constructor)(name, capacity)
}

pub fn get_object_converter() -> ObjectConverter {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    reg.object_converter.as_ref().unwrap().clone()
}

pub fn get_pyobject_converter() -> PyObjectConverter {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    reg.pyobject_converter.as_ref().unwrap().clone()
}
