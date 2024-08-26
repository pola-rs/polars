//! This is a heap allocated utility that can be used to register an object type.
//!
//! That object type will know its own generic type parameter `T` and callers can simply
//! send `&Any` values and don't have to know the generic type themselves.
use std::any::Any;
use std::fmt::{Debug, Formatter};
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use arrow::datatypes::ArrowDataType;
use once_cell::sync::Lazy;

use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::datatypes::AnyValue;
use crate::prelude::PolarsObject;
use crate::series::{IntoSeries, Series};

/// Takes a `name` and `capacity` and constructs a new builder.
pub type BuilderConstructor =
    Box<dyn Fn(&str, usize) -> Box<dyn AnonymousObjectBuilder> + Send + Sync>;
pub type ObjectConverter = Arc<dyn Fn(AnyValue) -> Box<dyn Any> + Send + Sync>;

pub struct ObjectRegistry {
    /// A function that creates an object builder
    pub builder_constructor: BuilderConstructor,
    // A function that converts AnyValue to Box<dyn Any> of the object type
    object_converter: Option<ObjectConverter>,
    pub physical_dtype: ArrowDataType,
}

impl Debug for ObjectRegistry {
    fn fmt(&self, f: &mut Formatter<'_>) -> std::fmt::Result {
        write!(f, "object-registry")
    }
}

impl ObjectRegistry {
    pub(super) fn new(
        builder_constructor: BuilderConstructor,
        physical_dtype: ArrowDataType,
    ) -> Self {
        Self {
            builder_constructor,
            object_converter: None,
            physical_dtype,
        }
    }
}

static GLOBAL_OBJECT_REGISTRY: Lazy<RwLock<Option<ObjectRegistry>>> = Lazy::new(Default::default);

/// This trait can be registered, after which that global registration
/// can be used to materialize object types
pub trait AnonymousObjectBuilder {
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
}

impl<T: PolarsObject> AnonymousObjectBuilder for ObjectChunkedBuilder<T> {
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
}

pub fn register_object_builder(
    builder_constructor: BuilderConstructor,
    object_converter: ObjectConverter,
    physical_dtype: ArrowDataType,
) {
    let reg = GLOBAL_OBJECT_REGISTRY.deref();
    let mut reg = reg.write().unwrap();

    *reg = Some(ObjectRegistry {
        builder_constructor,
        object_converter: Some(object_converter),
        physical_dtype,
    })
}

pub fn is_object_builder_registered() -> bool {
    let reg = GLOBAL_OBJECT_REGISTRY.deref();
    let reg = reg.read().unwrap();
    reg.is_some()
}

#[cold]
pub fn get_object_physical_type() -> ArrowDataType {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    reg.physical_dtype.clone()
}

pub fn get_object_builder(name: &str, capacity: usize) -> Box<dyn AnonymousObjectBuilder> {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    (reg.builder_constructor)(name, capacity)
}

pub fn get_object_converter() -> ObjectConverter {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    reg.object_converter.as_ref().unwrap().clone()
}
