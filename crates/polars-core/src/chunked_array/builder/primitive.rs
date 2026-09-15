use polars_array::PlPrimitiveArrayBuilder;

use super::*;

#[derive(Clone)]
pub struct PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    array_builder: PlPrimitiveArrayBuilder<T::Native>,
    pub(crate) field: Field,
}

impl<T> ChunkedBuilder<T::Native, T> for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: T::Native) {
        self.array_builder.push_value(v)
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.push_null()
    }

    fn finish(self) -> ChunkedArray<T> {
        let arr = self.array_builder.freeze().into_boxed();
        ChunkedArray::new_with_compute_len(Arc::new(self.field), vec![arr])
    }
}

impl<T> PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub fn new(name: PlSmallStr, capacity: usize) -> Self {
        PrimitiveChunkedBuilder {
            array_builder: PlPrimitiveArrayBuilder::<T::Native>::with_capacity(capacity),
            field: Field::new(name, T::get_static_dtype()),
        }
    }
}
