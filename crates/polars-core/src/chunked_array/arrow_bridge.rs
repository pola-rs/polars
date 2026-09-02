//! Handing a chunk to an Arrow kernel, and taking the result back.
//!
//! The chunks of a `ChunkedArray` are the arrays of `polars-array`; the kernels of
//! `polars-compute` are written against the Arrow arrays of `polars-arrow`. The two lay their
//! elements out the same way and are built on the same [`Buffer`](polars_buffer::Buffer) and
//! [`Bitmap`](arrow::bitmap::Bitmap), so crossing between them moves the backing buffers rather
//! than copying the elements.
//!
//! [`ToArrow`] is that crossing, typed: it names, for each array of `polars-array`, the Arrow
//! array that holds the same elements, so that a kernel written for one can be called on the
//! other without a downcast. It is defined on [`Flat`] arrays only — an Arrow array holds one slot
//! per element, which is what being flat means, and an array in the
//! [`scalar`](polars_array::broadcast) representation has to be written out before it can cross.
//! That is what keeps the cost visible at the call site: the dispatch in
//! [`arity`](crate::chunked_array::ops::arity) decides what a scalar chunk costs a kernel, and
//! reaches for a broadcasting kernel where the repeated value is all the kernel needs.

use arrow::array::{
    Array, BinaryArray, BinaryViewArray, BooleanArray, FixedSizeListArray, ListArray, NullArray,
    PrimitiveArray, StructArray, Utf8ViewArray,
};
use arrow::types::NativeType;
use polars_array::arrow::{export, import};
use polars_array::{
    Flat, PlArray, PlBinaryArray, PlBinaryViewArray, PlBooleanArray, PlFixedSizeListArray,
    PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray, PlUtf8ViewArray, StaticArray,
};

/// The Arrow array that holds the same elements as an array of `polars-array`.
///
/// See the [module docs](self) for what this is for and why it is defined on [`Flat`] arrays.
pub trait ToArrow: StaticArray {
    /// The Arrow array this one's buffers cross into.
    ///
    /// This is the Arrow array of the *physical* layout: a [`PlBinaryViewArray`] crosses into a
    /// [`BinaryViewArray`], and it is [`PlUtf8ViewArray`] — the wrapper that carries the UTF-8
    /// invariant — that crosses into a [`Utf8ViewArray`].
    type Arrow: Array;

    /// Hands the backing buffers of `array` to the Arrow array that holds the same elements.
    ///
    /// This is `O(1)`: `array` is flat, so every buffer already holds one slot per element.
    fn to_arrow(array: &Flat<Self>) -> Self::Arrow;

    /// Takes the backing buffers of `array` back.
    ///
    /// This is `O(1)`, and the result is flat, an Arrow array being one slot per element.
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
        // SAFETY: the elements of a `PlUtf8ViewArray` are valid UTF-8, which is the invariant the
        // wrapper carries.
        unsafe { export::binview_to_arrow_utf8view(array.as_binview()) }
    }

    #[inline]
    fn from_arrow(array: &Utf8ViewArray) -> Self {
        // SAFETY: the elements of a `Utf8ViewArray` are valid UTF-8.
        unsafe { PlUtf8ViewArray::from_binview_unchecked(import::binary_view_from_arrow(array)) }
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
///
/// This is [`ToArrow::to_arrow`] as a free function, for a caller that has the flatness already —
/// [`as_flat`] is what gets it. It is `O(1)`.
#[inline]
pub fn flat_to_arrow<A: ToArrow>(array: &Flat<A>) -> A::Arrow {
    A::to_arrow(array)
}

/// Takes the backing buffers of an Arrow array back as the chunk that holds the same elements.
///
/// This is [`ToArrow::from_arrow`] as a free function, which is what an Arrow kernel's result
/// crosses back through. It is `O(1)`, and the result is flat.
#[inline]
pub fn chunk_from_arrow<A: ToArrow>(array: &A::Arrow) -> A {
    A::from_arrow(array)
}

/// Hands `array` to the Arrow array that holds the same elements, writing it out first if it is
/// not laid out flat.
///
/// This is [`ToArrow::to_arrow`] for an array whose representation is not known: `O(1)` for a flat
/// array and `O(len)` for a scalar one. Prefer taking a [`Flat`] array where the caller can say
/// which it has.
#[inline]
pub fn chunk_to_arrow<A: ToArrow>(array: &A) -> A::Arrow {
    match array.as_flat() {
        Some(flat) => A::to_arrow(flat),
        None => A::to_arrow(&array.to_flat()),
    }
}

/// Borrows `array` as a flat one, writing out its buffers only if it is not laid out flat.
///
/// This is what a caller that needs a [`Flat`] array — to hand it to a kernel, or to read its
/// backing buffers — reaches for when it does not know the representation.
#[inline]
pub fn as_flat<A: StaticArray>(array: &A) -> std::borrow::Cow<'_, Flat<A>> {
    match array.as_flat() {
        Some(flat) => std::borrow::Cow::Borrowed(flat),
        None => std::borrow::Cow::Owned(array.to_flat()),
    }
}

/// Runs an Arrow kernel over a chunk of unknown type, importing what it hands back.
///
/// This is the untyped counterpart of [`ToArrow`], for the kernels that take a `dyn Array`. It is
/// `O(1)` at both ends for a flat chunk; a scalar one is written out by the export.
pub fn with_arrow_chunk<F>(chunk: &dyn PlArray, kernel: F) -> Box<dyn PlArray>
where
    F: FnOnce(&dyn Array) -> Box<dyn Array>,
{
    let arrow = export::to_arrow(chunk);
    import::from_arrow(&*kernel(&*arrow))
}
