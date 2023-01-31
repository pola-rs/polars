//! This is a heap allocated utility that can be used to register an object type.
//! That object type will know its own generic type parameter `T` and callers can simply
//! send `&Any` values and don't have to know the generic type themselves.
use std::any::Any;
use std::ops::Deref;
use std::sync::{Arc, RwLock};

use once_cell::sync::Lazy;

use crate::chunked_array::object::builder::ObjectChunkedBuilder;
use crate::datatypes::AnyValue;
use crate::prelude::PolarsObject;
use crate::series::{IntoSeries, Series};

pub type BuilderConstructor =
    Box<dyn Fn(&str, usize) -> Box<dyn AnonymousObjectBuilder> + Send + Sync>;
pub type ObjectConverter = Arc<dyn Fn(AnyValue) -> Box<dyn Any> + Send + Sync>;

struct GlobalObjectRegistry {
    /// A function that creates an object builder
    builder_constructor: BuilderConstructor,
    // A function that converts AnyValue to Box<dyn Any> of the object type
    object_converter: ObjectConverter,
}

static GLOBAL_OBJECT_REGISTRY: Lazy<RwLock<Option<GlobalObjectRegistry>>> =
    Lazy::new(Default::default);

pub trait AnonymousObjectBuilder {
    fn append_null(&mut self);

    fn append_value(&mut self, value: &dyn Any);

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
) {
    let reg = GLOBAL_OBJECT_REGISTRY.deref();
    let mut reg = reg.write().unwrap();

    *reg = Some(GlobalObjectRegistry {
        builder_constructor,
        object_converter,
    })
}

pub fn is_object_builder_registered() -> bool {
    let reg = GLOBAL_OBJECT_REGISTRY.deref();
    let reg = reg.read().unwrap();
    reg.is_some()
}

pub fn get_object_builder(name: &str, capacity: usize) -> Box<dyn AnonymousObjectBuilder> {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    (reg.builder_constructor)(name, capacity)
}

pub fn get_object_converter() -> ObjectConverter {
    let reg = GLOBAL_OBJECT_REGISTRY.read().unwrap();
    let reg = reg.as_ref().unwrap();
    reg.object_converter.clone()
}
