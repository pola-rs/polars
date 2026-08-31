pub use arrow::types::PrimitiveType;

/// The set of physical representations an array in this crate can have.
///
/// This is the counterpart of [`PhysicalType`](arrow::datatypes::PhysicalType): it has a
/// one-to-one mapping to each struct in this crate that implements [`PlArray`](crate::PlArray),
/// and is what [`PlArray::array_type`](crate::PlArray::array_type) hands out so that a `dyn PlArray` can be
/// downcast to a concrete array.
///
/// Unlike [`ArrowDataType`](arrow::datatypes::ArrowDataType), this carries no logical type — the
/// arrays in this crate are purely a physical representation, so there is nothing to distinguish a
/// timestamp from the `i64` it is stored as. It is also derived from the Rust type of the array
/// rather than stored in it, which is why [`PlArray::array_type`](crate::PlArray::array_type) returns it by
/// value and there is no way to change it.
///
/// # Example
/// ```
/// use polars_array::{PlArray, PlArrayType, PlBooleanArray, PlPrimitiveArray, PrimitiveType};
///
/// let arr: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3]));
/// assert_eq!(arr.array_type(), PlArrayType::Primitive(PrimitiveType::Int32));
/// assert!(arr.as_any().downcast_ref::<PlPrimitiveArray<i32>>().is_some());
///
/// let arr: Box<dyn PlArray> = Box::new(PlBooleanArray::from_vec(vec![true, false]));
/// assert_eq!(arr.array_type(), PlArrayType::Boolean);
/// ```
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlArrayType {
    /// A [`PlBooleanArray`](crate::PlBooleanArray): a boolean stored as a single bit.
    Boolean,
    /// A [`PlPrimitiveArray<T>`](crate::PlPrimitiveArray) where `T::PRIMITIVE` is this
    /// [`PrimitiveType`]: a value with a known compile-time size.
    Primitive(PrimitiveType),
}

impl PlArrayType {
    /// Whether this is [`PlArrayType::Primitive`] of type `primitive`.
    #[inline]
    pub fn eq_primitive(&self, primitive: PrimitiveType) -> bool {
        *self == Self::Primitive(primitive)
    }

    /// Whether this is [`PlArrayType::Primitive`] of any type.
    #[inline]
    pub fn is_primitive(&self) -> bool {
        matches!(self, Self::Primitive(_))
    }

    /// Whether this is [`PlArrayType::Boolean`].
    #[inline]
    pub fn is_boolean(&self) -> bool {
        matches!(self, Self::Boolean)
    }
}
