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
    /// A [`PlStructArray`](crate::PlStructArray): a row of one value per field array.
    ///
    /// The fields are part of neither this type nor the array's identity: two struct arrays are
    /// both [`PlArrayType::Struct`] no matter how many fields they have or what is in them.
    Struct,
    /// A [`PlListArray`](crate::PlListArray): a variable-length list of values.
    ///
    /// The values are part of neither this type nor the array's identity: two list arrays are both
    /// [`PlArrayType::List`] no matter what array their lists are taken over.
    List,
    /// A [`PlFixedSizeListArray`](crate::PlFixedSizeListArray): a list of a fixed number of
    /// values.
    ///
    /// Neither the values nor the width are part of this type or of the array's identity: two
    /// fixed size list arrays are both [`PlArrayType::FixedSizeList`] no matter how wide their
    /// lists are or what array they are taken over.
    FixedSizeList,
    /// A [`PlNullArray`](crate::PlNullArray): a null, with no value under it.
    Null,
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

    /// Whether this is [`PlArrayType::Struct`].
    #[inline]
    pub fn is_struct(&self) -> bool {
        matches!(self, Self::Struct)
    }

    /// Whether this is [`PlArrayType::List`].
    #[inline]
    pub fn is_list(&self) -> bool {
        matches!(self, Self::List)
    }

    /// Whether this is [`PlArrayType::FixedSizeList`].
    #[inline]
    pub fn is_fixed_size_list(&self) -> bool {
        matches!(self, Self::FixedSizeList)
    }

    /// Whether this is [`PlArrayType::Null`].
    #[inline]
    pub fn is_null(&self) -> bool {
        matches!(self, Self::Null)
    }
}
