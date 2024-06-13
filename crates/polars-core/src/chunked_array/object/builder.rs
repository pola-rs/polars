use super::*;
use crate::chunked_array::object::registry::{AnonymousObjectBuilder, ObjectRegistry};
use crate::utils::get_iter_capacity;

pub struct ObjectChunkedBuilder<T> {
    field: Field,
    bitmask_builder: MutableBitmap,
    values: Vec<T>,
}

impl<T> ObjectChunkedBuilder<T>
where
    T: PolarsObject,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        ObjectChunkedBuilder {
            field: Field::new(name, DataType::Object(T::type_name(), None)),
            values: Vec::with_capacity(capacity),
            bitmask_builder: MutableBitmap::with_capacity(capacity),
        }
    }

    /// Appends a value of type `T` into the builder
    #[inline]
    pub fn append_value(&mut self, v: T) {
        self.values.push(v);
        self.bitmask_builder.push(true);
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.values.push(T::default());
        self.bitmask_builder.push(false);
    }

    #[inline]
    pub fn append_value_from_any(&mut self, v: &dyn Any) -> PolarsResult<()> {
        let Some(v) = v.downcast_ref::<T>() else {
            polars_bail!(SchemaMismatch: "cannot downcast any in ObjectBuilder");
        };
        self.append_value(v.clone());
        Ok(())
    }

    #[inline]
    pub fn append_option(&mut self, opt: Option<T>) {
        match opt {
            Some(s) => self.append_value(s),
            None => self.append_null(),
        }
    }

    pub fn finish(mut self) -> ObjectChunked<T> {
        let null_bitmap: Option<Bitmap> = self.bitmask_builder.into();

        let len = self.values.len();
        let null_count = null_bitmap
            .as_ref()
            .map(|validity| validity.unset_bits())
            .unwrap_or(0) as IdxSize;

        let arr = Box::new(ObjectArray {
            values: Arc::new(self.values),
            null_bitmap,
            offset: 0,
            len,
        });

        self.field.dtype = get_object_type::<T>();

        unsafe {
            ChunkedArray::new_with_dims(Arc::new(self.field), vec![arr], len as IdxSize, null_count)
        }
    }
}

/// Initialize a polars Object data type. The type has got information needed to
/// construct new objects.
pub(crate) fn get_object_type<T: PolarsObject>() -> DataType {
    let object_builder = Box::new(|name: &str, capacity: usize| {
        Box::new(ObjectChunkedBuilder::<T>::new(name, capacity)) as Box<dyn AnonymousObjectBuilder>
    });

    let object_size = std::mem::size_of::<T>();
    let physical_dtype = ArrowDataType::FixedSizeBinary(object_size);

    let registry = ObjectRegistry::new(object_builder, physical_dtype);
    DataType::Object(T::type_name(), Some(Arc::new(registry)))
}

impl<T> Default for ObjectChunkedBuilder<T>
where
    T: PolarsObject,
{
    fn default() -> Self {
        ObjectChunkedBuilder::new("", 0)
    }
}

impl<T> NewChunkedArray<ObjectType<T>, T> for ObjectChunked<T>
where
    T: PolarsObject,
{
    fn from_slice(name: &str, v: &[T]) -> Self {
        Self::from_iter_values(name, v.iter().cloned())
    }

    fn from_slice_options(name: &str, opt_v: &[Option<T>]) -> Self {
        let mut builder = ObjectChunkedBuilder::<T>::new(name, opt_v.len());
        opt_v
            .iter()
            .cloned()
            .for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    fn from_iter_options(name: &str, it: impl Iterator<Item = Option<T>>) -> ObjectChunked<T> {
        let mut builder = ObjectChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|opt| builder.append_option(opt));
        builder.finish()
    }

    /// Create a new ChunkedArray from an iterator.
    fn from_iter_values(name: &str, it: impl Iterator<Item = T>) -> ObjectChunked<T> {
        let mut builder = ObjectChunkedBuilder::new(name, get_iter_capacity(&it));
        it.for_each(|v| builder.append_value(v));
        builder.finish()
    }
}

impl<T> ObjectChunked<T>
where
    T: PolarsObject,
{
    pub fn new_from_vec(name: &str, v: Vec<T>) -> Self {
        let field = Arc::new(Field::new(name, DataType::Object(T::type_name(), None)));
        let len = v.len();
        let arr = Box::new(ObjectArray {
            values: Arc::new(v),
            null_bitmap: None,
            offset: 0,
            len,
        });

        unsafe { ObjectChunked::new_with_dims(field, vec![arr], len as IdxSize, 0) }
    }

    pub fn new_from_vec_and_validity(name: &str, v: Vec<T>, validity: Bitmap) -> Self {
        let field = Arc::new(Field::new(name, DataType::Object(T::type_name(), None)));
        let len = v.len();
        let null_count = validity.unset_bits();
        let arr = Box::new(ObjectArray {
            values: Arc::new(v),
            null_bitmap: Some(validity),
            offset: 0,
            len,
        });

        unsafe {
            ObjectChunked::new_with_dims(field, vec![arr], len as IdxSize, null_count as IdxSize)
        }
    }

    pub fn new_empty(name: &str) -> Self {
        Self::new_from_vec(name, vec![])
    }
}

/// Convert a Series of dtype object to an Arrow Array of FixedSizeBinary
pub(crate) fn object_series_to_arrow_array(s: &Series) -> ArrayRef {
    // The list builder knows how to create an arrow array
    // we simply piggy back on that code.

    // SAFETY: 0..len is in bounds
    let list_s = unsafe {
        s.agg_list(&GroupsProxy::Slice {
            groups: vec![[0, s.len() as IdxSize]],
            rolling: false,
        })
    };
    let arr = &list_s.chunks()[0];
    let arr = arr.as_any().downcast_ref::<ListArray<i64>>().unwrap();
    arr.values().to_boxed()
}
