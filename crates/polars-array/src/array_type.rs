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
    ///
    /// The offsets are part of neither this type nor the array's identity: two binary arrays are
    /// both [`PlArrayType::Binary`] no matter how their bytes are cut into elements. Nothing here
    /// says the bytes are a string either — that is a logical type, which the arrays in this crate
    /// do not carry.
    Binary,
    /// A [`PlBinaryViewArray`](crate::PlBinaryViewArray): a variable-length sequence of bytes,
    /// stored as a view that either inlines them or points at a data buffer.
    ///
    /// The data buffers are part of neither this type nor the array's identity: two binary view
    /// arrays are both [`PlArrayType::BinaryView`] no matter how the bytes of their elements are
    /// reached. Nothing here says the bytes are a string either — that is
    /// [`PlArrayType::Utf8View`], the one logical promise these arrays do carry.
    BinaryView,
    /// A [`PlUtf8ViewArray`](crate::PlUtf8ViewArray): a [`PlArrayType::BinaryView`] whose bytes
    /// are known to be valid UTF-8.
    ///
    /// This is the one place these arrays carry a logical type, and it is what makes a string
    /// array distinguishable from the byte array it is stored as: a `dyn PlArray` of
    /// [`PlArrayType::Utf8View`] downcasts to a [`PlUtf8ViewArray`](crate::PlUtf8ViewArray) and
    /// exports as an Arrow [`Utf8ViewArray`](arrow::array::Utf8ViewArray), where a
    /// [`PlArrayType::BinaryView`] does neither. See [`crate::utf8view`] for why that one
    /// invariant is worth a type of its own.
    Utf8View,
    /// A [`PlFixedSizeBinaryArray`](crate::PlFixedSizeBinaryArray): a sequence of bytes of a fixed
    /// width, stored in one values buffer the elements cut into consecutive slices.
    ///
    /// The width is part of neither this type nor the array's identity: two fixed size binary
    /// arrays are both [`PlArrayType::FixedSizeBinary`] no matter how wide their elements are.
    /// Nothing here says the bytes are a decimal either — that is a logical type, which the arrays
    /// in this crate do not carry.
    FixedSizeBinary,
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
    ///
    /// This is `false` for [`PlArrayType::Utf8View`], which is a distinct array type even though
    /// it is stored as a binary view — see [`Self::is_view`] for the test that spans both.
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
