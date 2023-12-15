use super::*;

#[derive(Clone)]
pub struct PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    array_builder: MutablePrimitiveArray<T::Native>,
    pub(crate) field: Field,
}

impl<T> ChunkedBuilder<T::Native, T> for PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    /// Appends a value of type `T` into the builder
    #[inline]
    fn append_value(&mut self, v: T::Native) {
        self.array_builder.push(Some(v))
    }

    /// Appends a null slot into the builder
    #[inline]
    fn append_null(&mut self) {
        self.array_builder.push(None)
    }

    fn finish(mut self) -> ChunkedArray<T> {
        let arr = self.array_builder.as_box();
        let mut ca = ChunkedArray {
            field: Arc::new(self.field),
            chunks: vec![arr],
            phantom: PhantomData,
            bit_settings: Default::default(),
            length: 0,
            null_count: 0,
        };
        ca.compute_len();
        ca
    }

    fn shrink_to_fit(&mut self) {
        self.array_builder.shrink_to_fit()
    }
}

impl<T> PrimitiveChunkedBuilder<T>
where
    T: PolarsNumericType,
{
    pub fn new(name: &str, capacity: usize) -> Self {
        let array_builder = MutablePrimitiveArray::<T::Native>::with_capacity(capacity)
            .to(T::get_dtype().to_arrow());

        PrimitiveChunkedBuilder {
            array_builder,
            field: Field::new(name, T::get_dtype()),
        }
    }
}
