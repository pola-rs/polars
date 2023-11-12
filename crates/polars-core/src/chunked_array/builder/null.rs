use arrow::legacy::array::null::MutableNullArray;

use super::*;
use crate::series::implementations::null::NullChunked;

#[derive(Clone)]
pub struct NullChunkedBuilder {
    array_builder: MutableNullArray,
    pub(crate) field: Field,
}

impl NullChunkedBuilder {
    pub fn new(name: &str, len: usize) -> Self {
        let array_builder = MutableNullArray::new(len);

        NullChunkedBuilder {
            array_builder,
            field: Field::new(name, DataType::Null),
        }
    }

    /// Appends a null slot into the builder
    #[inline]
    pub fn append_null(&mut self) {
        self.array_builder.push_null()
    }

    pub fn finish(mut self) -> NullChunked {
        let arr = self.array_builder.as_box();
        let ca = NullChunked::new(Arc::from(self.field.name.as_str()), arr.len());
        ca
    }

    pub fn shrink_to_fit(&mut self) {
        self.array_builder.shrink_to_fit()
    }
}
