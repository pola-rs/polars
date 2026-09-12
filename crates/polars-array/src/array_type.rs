pub use arrow::types::PrimitiveType;

/// The set of physical representations an array in this crate can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlArrayType {
    /// A [`PlBooleanArray`](crate::PlBooleanArray): a boolean stored as a single bit.
    Boolean,
    /// A [`PlPrimitiveArray<T>`](crate::PlPrimitiveArray): a value with a known compile-time size.
    Primitive(PrimitiveType),
    /// A [`PlBinaryArray`](crate::PlBinaryArray): bytes cut out of one values buffer by offsets.
    Binary,
    /// A [`PlBinaryViewArray`](crate::PlBinaryViewArray): bytes inlined in a view or pointed at.
    BinaryView,
    /// A [`PlUtf8ViewArray`](crate::PlUtf8ViewArray): a `BinaryView` known to be valid UTF-8.
    Utf8View,
    /// A [`PlFixedSizeBinaryArray`](crate::PlFixedSizeBinaryArray): bytes of a fixed width.
    FixedSizeBinary,
    /// A [`PlStructArray`](crate::PlStructArray): a row of one value per field array.
    Struct,
    /// A [`PlListArray`](crate::PlListArray): a variable-length list of values.
    List,
    /// A [`PlFixedSizeListArray`](crate::PlFixedSizeListArray): a list of a fixed number of values.
    FixedSizeList,
    /// A [`PlNullArray`](crate::PlNullArray): a null, with no value under it.
    Null,
    /// PolarsObject.
    Object { type_name: &'static str },
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

    /// Whether this is [`PlArrayType::Binary`].
    #[inline]
    pub fn is_binary(&self) -> bool {
        matches!(self, Self::Binary)
    }

    /// Whether this is [`PlArrayType::BinaryView`].
    #[inline]
    pub fn is_binary_view(&self) -> bool {
        matches!(self, Self::BinaryView)
    }

    /// Whether this is [`PlArrayType::Utf8View`].
    #[inline]
    pub fn is_utf8_view(&self) -> bool {
        matches!(self, Self::Utf8View)
    }

    /// Whether this is [`PlArrayType::BinaryView`] or [`PlArrayType::Utf8View`], the view types.
    #[inline]
    pub fn is_view(&self) -> bool {
        matches!(self, Self::BinaryView | Self::Utf8View)
    }

    /// Whether this is [`PlArrayType::FixedSizeBinary`].
    #[inline]
    pub fn is_fixed_size_binary(&self) -> bool {
        matches!(self, Self::FixedSizeBinary)
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
