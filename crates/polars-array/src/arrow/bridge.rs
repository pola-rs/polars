//! Handing a chunk to an Arrow kernel, and taking the result back.

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray, ListArray, NullArray,
    PrimitiveArray, StructArray, Utf8ViewArray,
};
use arrow::types::NativeType;

use crate::arrow::{export, import};
use crate::{
    Flat, PlArray, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeListArray,
    PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray, PlUtf8ViewArray, StaticArray,
};

/// The Arrow array that holds the same elements as an array of this crate.
pub trait ToArrow: StaticArray {
    /// The Arrow array this one's buffers cross into.
    type Arrow: Array;

    /// Hands the backing buffers of `array` to the Arrow array that holds the same elements.
    fn to_arrow(array: &Flat<Self>) -> Self::Arrow;

    /// Takes the backing buffers of `array` back.
    fn from_arrow(array: &Self::Arrow) -> Self;
}

impl<T: NativeType> ToArrow for PlPrimitiveArray<T> {
    type Arrow = PrimitiveArray<T>;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> PrimitiveArray<T> {
        export::primitive_to_arrow_primitive(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &PrimitiveArray<T>) -> Self {
        import::primitive_from_arrow(array)
    }
}

impl ToArrow for PlBooleanArray {
    type Arrow = BooleanArray;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> BooleanArray {
        export::boolean_to_arrow_boolean(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &BooleanArray) -> Self {
        import::boolean_from_arrow(array)
    }
}

impl ToArrow for PlBinaryViewArray {
    type Arrow = BinaryViewArray;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> BinaryViewArray {
        export::binview_to_arrow_binview(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &BinaryViewArray) -> Self {
        import::binary_view_from_arrow(array)
    }
}

impl ToArrow for PlUtf8ViewArray {
    type Arrow = Utf8ViewArray;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> Utf8ViewArray {
        export::utf8view_to_arrow_utf8view(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &Utf8ViewArray) -> Self {
        import::utf8_view_from_arrow(array)
    }
}

impl ToArrow for PlBinaryArray {
    type Arrow = BinaryArray<i64>;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> BinaryArray<i64> {
        export::binary_to_arrow_large_binary(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &BinaryArray<i64>) -> Self {
        import::binary_from_arrow(array)
    }
}

impl ToArrow for PlListArray {
    type Arrow = ListArray<i64>;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> ListArray<i64> {
        export::list_to_arrow_large_list(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &ListArray<i64>) -> Self {
        import::list_from_arrow(array)
    }
}

impl ToArrow for PlFixedSizeListArray {
    type Arrow = FixedSizeListArray;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> FixedSizeListArray {
        export::fixed_size_list_to_arrow_fixed_size_list(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &FixedSizeListArray) -> Self {
        import::fixed_size_list_from_arrow(array)
    }
}

impl ToArrow for PlStructArray {
    type Arrow = StructArray;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> StructArray {
        export::struct_to_arrow_struct(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &StructArray) -> Self {
        import::struct_from_arrow(array)
    }
}

impl ToArrow for PlNullArray {
    type Arrow = NullArray;

    #[inline]
    fn to_arrow(array: &Flat<Self>) -> NullArray {
        export::null_to_arrow_null(array.as_array())
    }

    #[inline]
    fn from_arrow(array: &NullArray) -> Self {
        import::null_from_arrow(array)
    }
}

/// Hands the backing buffers of a flat chunk to the Arrow array that holds the same elements.
#[inline]
pub fn flat_to_arrow<A: ToArrow>(array: &Flat<A>) -> A::Arrow {
    A::to_arrow(array)
}

/// Takes the backing buffers of an Arrow array back as the chunk that holds the same elements.
#[inline]
pub fn chunk_from_arrow<A: ToArrow>(array: &A::Arrow) -> A {
    A::from_arrow(array)
}

/// Hands `array` to the Arrow array that holds the same elements, writing it out first if it is not
/// laid out flat.
#[inline]
pub fn chunk_to_arrow<A: ToArrow>(array: &A) -> A::Arrow {
    match array.as_flat() {
        Some(flat) => A::to_arrow(flat),
        None => A::to_arrow(&array.to_flat()),
    }
}

/// Runs an Arrow kernel over a chunk of unknown type, importing what it hands back.
pub fn with_arrow_chunk<F>(chunk: &dyn PlArray, kernel: F) -> Box<dyn PlArray>
where
    F: FnOnce(&dyn Array) -> Box<dyn Array>,
{
    let arrow = export::to_arrow(chunk);
    import::from_arrow(&*kernel(&*arrow))
}
