pub use arrow::types::PrimitiveType;

/// The set of physical representations an array in this crate can have.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
pub enum PlArrayType {
    /// A [`PlBooleanArray`](crate::PlBooleanArray): a boolean stored as a single bit.
    Boolean,
    /// A [`PlPrimitiveArray<T>`](crate::PlPrimitiveArray) where `T::PRIMITIVE` is this
    /// [`PrimitiveType`]: a value with a known compile-time size.
    Primitive(PrimitiveType),
    /// A [`PlBinaryArray`](crate::PlBinaryArray): a variable-length sequence of bytes, stored as
    /// the offsets that cut one values buffer into consecutive slices.
    Binary,
    /// A [`PlBinaryViewArray`](crate::PlBinaryViewArray): a variable-length sequence of bytes,
    /// stored as a view that either inlines them or points at a data buffer.
    BinaryView,
    /// A [`PlUtf8ViewArray`](crate::PlUtf8ViewArray): a [`PlArrayType::BinaryView`] whose bytes are
    /// known to be valid UTF-8.
    Utf8View,
    /// A [`PlFixedSizeBinaryArray`](crate::PlFixedSizeBinaryArray): a sequence of bytes of a fixed
    /// width, stored in one values buffer the elements cut into consecutive slices.
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

    /// Whether this is [`PlArrayType::BinaryView`] or [`PlArrayType::Utf8View`], the two array
    /// types stored as a view over a set of data buffers.
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
