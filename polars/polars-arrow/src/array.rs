use crate::vec::AlignedVec;
use arrow::array::{Array, ArrayDataRef, ArrayRef, BooleanBufferBuilder, PrimitiveArray};
use arrow::datatypes::ArrowPrimitiveType;
use num::Num;

pub trait GetValues {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num;
}

impl GetValues for ArrayDataRef {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        debug_assert_eq!(&T::DATA_TYPE, self.data_type());
        // the first buffer is the value array
        let value_buf = &self.buffers()[0];
        let offset = self.offset();
        let vals = unsafe { value_buf.typed_data::<T::Native>() };
        &vals[offset..offset + self.len()]
    }
}

impl GetValues for &dyn Array {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        self.data_ref().get_values::<T>()
    }
}

impl GetValues for ArrayRef {
    fn get_values<T>(&self) -> &[T::Native]
    where
        T: ArrowPrimitiveType,
        T::Native: Num,
    {
        self.data_ref().get_values::<T>()
    }
}

pub trait ToPrimitive {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType;
}

impl ToPrimitive for ArrayDataRef {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
    {
        PrimitiveArray::from(self)
    }
}

impl ToPrimitive for &dyn Array {
    fn into_primitive_array<T>(self) -> PrimitiveArray<T>
    where
        T: ArrowPrimitiveType,
    {
        self.data().into_primitive_array()
    }
}

/// An arrow primitive builder that is faster than Arrow's native builder because it uses Rust Vec's
/// as buffer
pub struct PrimitiveArrayBuilder<T>
where
    T: ArrowPrimitiveType,
    T::Native: Default,
{
    values: AlignedVec<T::Native>,
    bitmap_builder: BooleanBufferBuilder,
    null_count: usize,
}

impl<T> PrimitiveArrayBuilder<T>
where
    T: ArrowPrimitiveType,
    T::Native: Default,
{
    pub fn new(capacity: usize) -> Self {
        let values = AlignedVec::<T::Native>::with_capacity_aligned(capacity);
        let bitmap_builder = BooleanBufferBuilder::new(capacity);

        Self {
            values,
            bitmap_builder,
            null_count: 0,
        }
    }

    /// Appends a value of type `T::Native` into the builder
    #[inline]
    pub fn append_value(&mut self, v: T::Native) {
        self.values.push(v);
        self.bitmap_builder.append(true);
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.bitmap_builder.append(false);
        self.values.push(Default::default());
        self.null_count += 1;
    }

    pub fn finish(mut self) -> PrimitiveArray<T> {
        let null_bit_buffer = self.bitmap_builder.finish();
        let buf = if self.null_count == 0 {
            None
        } else {
            Some(null_bit_buffer)
        };
        self.values.into_primitive_array(buf)
    }
}
