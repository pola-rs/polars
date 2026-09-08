//! Building arrays element by element, or array by array.

/// Whether a builder may adopt the buffers of the arrays it appends, rather than copy them.
pub use arrow::array::builder::ShareStrategy;
use arrow::bitmap::OptBitmapBuilder;
use arrow::types::{days_ms, i256, months_days_ns};
use polars_utils::IdxSize;
use polars_utils::float16::pf16;
use polars_utils::index::ChunkId;

use crate::array::PlArray;
use crate::array_type::{PlArrayType, PrimitiveType};
use crate::bitmap::PlBitmapRef;
use crate::static_array::StaticArray;
use crate::{
    PlBinaryArray, PlBinaryArrayBuilder, PlBinaryViewArray, PlBinaryViewArrayBuilder,
    PlBooleanArray, PlBooleanArrayBuilder, PlFixedSizeBinaryArray, PlFixedSizeBinaryArrayBuilder,
    PlFixedSizeListArray, PlFixedSizeListArrayBuilder, PlListArray, PlListArrayBuilder,
    PlNullArray, PlNullArrayBuilder, PlPrimitiveArray, PlPrimitiveArrayBuilder, PlStructArray,
    PlStructArrayBuilder, PlUtf8ViewArray, PlUtf8ViewArrayBuilder,
    with_match_pl_primitive_array_type,
};

/// A builder of one concrete array type.
pub trait StaticArrayBuilder: Send {
    /// The array this builder builds.
    type Array: StaticArray;

    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// The number of elements appended so far.
    fn len(&self) -> usize;

    /// Whether no element has been appended yet.
    #[inline]
    fn is_empty(&self) -> bool {
        self.len() == 0
    }

    /// Consumes this builder, returning the array it built.
    fn freeze(self) -> Self::Array;

    /// Returns the array this builder built, leaving it empty and ready to build another.
    fn freeze_reset(&mut self) -> Self::Array;

    /// Appends `length` nulls.
    fn extend_nulls(&mut self, length: usize);

    /// Appends every element of `other`, in order.
    #[inline]
    fn extend(&mut self, other: &Self::Array, share: ShareStrategy) {
        self.subslice_extend(other, 0, other.len(), share);
    }

    /// Appends the `length` elements of `other` starting at `start`, in order.
    fn subslice_extend(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    );

    /// Appends the single element of `other` at `index`.
    ///
    /// # Safety
    /// `index` must be smaller than `other.len()`.
    #[inline]
    unsafe fn extend_one(&mut self, other: &Self::Array, index: usize, share: ShareStrategy) {
        self.subslice_extend(other, index, 1, share);
    }

    /// Appends the `length` elements of `other` starting at `start` `repeats` times over.
    fn subslice_extend_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        self.reserve(length * repeats);
        for _ in 0..repeats {
            self.subslice_extend(other, start, length, share);
        }
    }

    /// Appends each of the `length` elements of `other` starting at `start` `repeats` times over.
    fn subslice_extend_each_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Appends the elements `ids` names, in order, out of the chunks of one chunked array.
    ///
    /// # Safety
    /// Every id must name a chunk of `chunks` and an element of that chunk.
    unsafe fn chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&Self::Array],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        self.reserve(ids.len());

        // One chunk is the common case, and it lets the chunk lookup leave the loop.
        if let [chunk] = chunks {
            for id in ids {
                let (_, array_idx) = id.extract();
                // SAFETY: the caller guarantees the id names an element of the chunk.
                unsafe { self.extend_one(chunk, array_idx as usize, share) };
            }
            return;
        }

        for id in ids {
            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: the caller guarantees the id names a chunk and an element of it.
            unsafe {
                let chunk = chunks.get_unchecked(chunk_idx as usize);
                self.extend_one(chunk, array_idx as usize, share);
            }
        }
    }

    /// Appends the elements `ids` names out of the chunks, a null id standing for a null element.
    ///
    /// # Safety
    /// Every id that is not null must name a chunk of `chunks` and an element of that chunk.
    unsafe fn opt_chunked_gather_extend<const B: u64>(
        &mut self,
        chunks: &[&Self::Array],
        ids: &[ChunkId<B>],
        share: ShareStrategy,
    ) {
        self.reserve(ids.len());

        for id in ids {
            if id.is_null() {
                self.extend_nulls(1);
                continue;
            }

            let (chunk_idx, array_idx) = id.extract();
            // SAFETY: the id is not null, so the caller guarantees it names an element.
            unsafe {
                let chunk = chunks.get_unchecked(chunk_idx as usize);
                self.extend_one(chunk, array_idx as usize, share);
            }
        }
    }

    /// Appends the element of `other` at every index of `idxs`, in the order they are given.
    ///
    /// # Safety
    /// Every index must be smaller than `other.len()`.
    unsafe fn gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy);

    /// Appends the element of `other` at every index of `idxs`; an out-of-bounds index is a null.
    fn opt_gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy);
}

/// A trait object over the builders in this crate.
#[allow(private_bounds)]
pub trait PlArrayBuilder: PlArrayBuilderBoxedHelper + Send {
    /// Reserves capacity for at least `additional` more elements.
    fn reserve(&mut self, additional: usize);

    /// The number of elements appended so far.
    fn len(&self) -> usize;

    /// Whether no element has been appended yet.
    fn is_empty(&self) -> bool;

    /// Consumes this builder, returning the array it built.
    fn freeze(self) -> Box<dyn PlArray>;

    /// Returns the array this builder built, leaving it empty and ready to build another.
    fn freeze_reset(&mut self) -> Box<dyn PlArray>;

    /// Appends `length` nulls.
    fn extend_nulls(&mut self, length: usize);

    /// Appends every element of `other`, in order.
    fn extend(&mut self, other: &dyn PlArray, share: ShareStrategy);

    /// Appends the `length` elements of `other` starting at `start`, in order.
    fn subslice_extend(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    );

    /// Appends the `length` elements of `other` starting at `start` `repeats` times over.
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Appends each of the `length` elements of `other` starting at `start` `repeats` times over.
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Appends the element of `other` at every index of `idxs`, in the order they are given.
    ///
    /// # Safety
    /// Every index must be smaller than `other.len()`.
    unsafe fn gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy);

    /// Appends the element of `other` at every index of `idxs`; an out-of-bounds index is a null.
    fn opt_gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy);
}

/// The [`PlArrayBuilder::freeze`] of a boxed builder, which is the form a trait object admits.
trait PlArrayBuilderBoxedHelper {
    fn freeze_boxed(self: Box<Self>) -> Box<dyn PlArray>;
}

impl<B: PlArrayBuilder> PlArrayBuilderBoxedHelper for B {
    #[inline]
    fn freeze_boxed(self: Box<Self>) -> Box<dyn PlArray> {
        (*self).freeze()
    }
}

/// Downcasts `array` to the array `B` builds.
#[inline]
fn downcast<B: StaticArrayBuilder>(array: &dyn PlArray) -> &B::Array {
    array
        .as_any()
        .downcast_ref::<B::Array>()
        .unwrap_or_else(|| {
            panic!(
                "cannot append a {:?} array to a builder of a different array type",
                array.array_type(),
            )
        })
}

impl<B: StaticArrayBuilder> PlArrayBuilder for B {
    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        StaticArrayBuilder::reserve(self, additional);
    }

    #[inline(always)]
    fn len(&self) -> usize {
        StaticArrayBuilder::len(self)
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        StaticArrayBuilder::is_empty(self)
    }

    #[inline(always)]
    fn freeze(self) -> Box<dyn PlArray> {
        Box::new(StaticArrayBuilder::freeze(self))
    }

    #[inline(always)]
    fn freeze_reset(&mut self) -> Box<dyn PlArray> {
        Box::new(StaticArrayBuilder::freeze_reset(self))
    }

    #[inline(always)]
    fn extend_nulls(&mut self, length: usize) {
        StaticArrayBuilder::extend_nulls(self, length);
    }

    #[inline(always)]
    fn extend(&mut self, other: &dyn PlArray, share: ShareStrategy) {
        StaticArrayBuilder::extend(self, downcast::<Self>(other), share);
    }

    #[inline(always)]
    fn subslice_extend(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        StaticArrayBuilder::subslice_extend(self, downcast::<Self>(other), start, length, share);
    }

    #[inline(always)]
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        StaticArrayBuilder::subslice_extend_repeated(
            self,
            downcast::<Self>(other),
            start,
            length,
            repeats,
            share,
        );
    }

    #[inline(always)]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        StaticArrayBuilder::subslice_extend_each_repeated(
            self,
            downcast::<Self>(other),
            start,
            length,
            repeats,
            share,
        );
    }

    #[inline(always)]
    unsafe fn gather_extend(
        &mut self,
        other: &dyn PlArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the indices are in bounds of the array, which is the one downcast here.
        unsafe { StaticArrayBuilder::gather_extend(self, downcast::<Self>(other), idxs, share) };
    }

    #[inline(always)]
    fn opt_gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy) {
        StaticArrayBuilder::opt_gather_extend(self, downcast::<Self>(other), idxs, share);
    }
}

/// A boxed builder is a builder: this is what lets the nested builders be built out of one.
impl PlArrayBuilder for Box<dyn PlArrayBuilder> {
    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        (**self).reserve(additional);
    }

    #[inline(always)]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline(always)]
    fn is_empty(&self) -> bool {
        (**self).is_empty()
    }

    #[inline(always)]
    fn freeze(self) -> Box<dyn PlArray> {
        self.freeze_boxed()
    }

    #[inline(always)]
    fn freeze_reset(&mut self) -> Box<dyn PlArray> {
        (**self).freeze_reset()
    }

    #[inline(always)]
    fn extend_nulls(&mut self, length: usize) {
        (**self).extend_nulls(length);
    }

    #[inline(always)]
    fn extend(&mut self, other: &dyn PlArray, share: ShareStrategy) {
        (**self).extend(other, share);
    }

    #[inline(always)]
    fn subslice_extend(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend(other, start, length, share);
    }

    #[inline(always)]
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend_repeated(other, start, length, repeats, share);
    }

    #[inline(always)]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn PlArray,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend_each_repeated(other, start, length, repeats, share);
    }

    #[inline(always)]
    unsafe fn gather_extend(
        &mut self,
        other: &dyn PlArray,
        idxs: &[IdxSize],
        share: ShareStrategy,
    ) {
        // SAFETY: the indices are in bounds of `other`, which this only passes on.
        unsafe { (**self).gather_extend(other, idxs, share) };
    }

    #[inline(always)]
    fn opt_gather_extend(&mut self, other: &dyn PlArray, idxs: &[IdxSize], share: ShareStrategy) {
        (**self).opt_gather_extend(other, idxs, share);
    }
}

/// Calls `extend` once per maximal run of consecutive indices in `idxs`, with the index the run
/// starts at and how many indices it covers.
pub(crate) fn for_each_run(idxs: &[IdxSize], mut extend: impl FnMut(usize, usize)) {
    let mut run_start = 0;

    while run_start < idxs.len() {
        let first = idxs[run_start] as usize;
        let mut run_length = 1;

        while run_start + run_length < idxs.len()
            && idxs[run_start + run_length] as usize == first + run_length
        {
            run_length += 1;
        }

        extend(first, run_length);
        run_start += run_length;
    }
}

/// An empty builder of the arrays that `array` is one of.
pub fn builder_like(array: &dyn PlArray) -> Box<dyn PlArrayBuilder> {
    match array.array_type() {
        PlArrayType::Primitive(_) => with_match_pl_primitive_array_type!(array, |T| {
            Box::new(PlPrimitiveArrayBuilder::<T>::new()) as Box<dyn PlArrayBuilder>
        })
        .expect("a primitive array has a primitive element type"),
        PlArrayType::Boolean => Box::new(PlBooleanArrayBuilder::new()),
        PlArrayType::Binary => Box::new(PlBinaryArrayBuilder::new()),
        PlArrayType::BinaryView => Box::new(PlBinaryViewArrayBuilder::new()),
        PlArrayType::Utf8View => Box::new(PlUtf8ViewArrayBuilder::new()),
        PlArrayType::FixedSizeBinary => {
            let array = array
                .as_any()
                .downcast_ref::<PlFixedSizeBinaryArray>()
                .unwrap();
            Box::new(PlFixedSizeBinaryArrayBuilder::new(array.width()))
        },
        PlArrayType::Null => Box::new(PlNullArrayBuilder::new()),
        PlArrayType::List => {
            let array = array.as_any().downcast_ref::<PlListArray>().unwrap();
            Box::new(PlListArrayBuilder::new(builder_like(array.values())))
        },
        PlArrayType::FixedSizeList => {
            let array = array
                .as_any()
                .downcast_ref::<PlFixedSizeListArray>()
                .unwrap();
            // The values hold the values the elements are made of in either representation,
            // which is all a builder for them is taken from.
            Box::new(PlFixedSizeListArrayBuilder::new(
                builder_like(array.values()),
                array.width(),
            ))
        },
        PlArrayType::Struct => {
            let array = array.as_any().downcast_ref::<PlStructArray>().unwrap();
            let fields = array
                .fields()
                .iter()
                .map(|field| builder_like(&**field))
                .collect();
            Box::new(PlStructArrayBuilder::new(fields))
        },
        x @ PlArrayType::Object { .. } => {
            panic!("polars-array: no PlArrayBuilder for {x:?}")
        },
    }
}

/// An array of `length` nulls shaped like `arr`, in `O(1)` memory.
pub fn new_full_null_like(arr: &dyn PlArray, length: usize) -> Box<dyn PlArray> {
    arr.new_full_null_like_self(length)
}

/// An array of `length` nulls of the array type `array_type` names, in `O(1)` memory.
///
/// # Panics
/// For [`PlArrayType::Object`], whose rust type an array type does not name.
pub fn new_full_null(array_type: PlArrayType, length: usize) -> Box<dyn PlArray> {
    macro_rules! new_full_null {
        ($A:ty) => {
            Box::new(<$A as StaticArray>::new_full_null(length)) as Box<dyn PlArray>
        };
    }

    match array_type {
        PlArrayType::Null => new_full_null!(PlNullArray),
        PlArrayType::Boolean => new_full_null!(PlBooleanArray),
        PlArrayType::Primitive(primitive) => primitive_new_full_null(primitive, length),
        PlArrayType::Binary => new_full_null!(PlBinaryArray),
        PlArrayType::BinaryView => new_full_null!(PlBinaryViewArray),
        PlArrayType::Utf8View => new_full_null!(PlUtf8ViewArray),
        PlArrayType::FixedSizeBinary => new_full_null!(PlFixedSizeBinaryArray),
        PlArrayType::List => new_full_null!(PlListArray),
        PlArrayType::FixedSizeList => new_full_null!(PlFixedSizeListArray),
        PlArrayType::Struct => new_full_null!(PlStructArray),
        x @ PlArrayType::Object { .. } => {
            panic!("polars-array: cannot build a full null {x:?} typed array")
        },
    }
}

/// A [`PlPrimitiveArray`] of `length` nulls over the elements `primitive` names.
fn primitive_new_full_null(primitive: PrimitiveType, length: usize) -> Box<dyn PlArray> {
    macro_rules! new_full_null {
        ($T:ty) => {
            Box::new(PlPrimitiveArray::<$T>::new_full_null(length)) as Box<dyn PlArray>
        };
    }

    match primitive {
        PrimitiveType::Int8 => new_full_null!(i8),
        PrimitiveType::Int16 => new_full_null!(i16),
        PrimitiveType::Int32 => new_full_null!(i32),
        PrimitiveType::Int64 => new_full_null!(i64),
        PrimitiveType::Int128 => new_full_null!(i128),
        PrimitiveType::Int256 => new_full_null!(i256),
        PrimitiveType::UInt8 => new_full_null!(u8),
        PrimitiveType::UInt16 => new_full_null!(u16),
        PrimitiveType::UInt32 => new_full_null!(u32),
        PrimitiveType::UInt64 => new_full_null!(u64),
        // A `View` and a `u128` are both `PrimitiveType::UInt128`, which therefore does not pin
        // the element type down: an array of views is asked for by the array that holds one.
        PrimitiveType::UInt128 => new_full_null!(u128),
        PrimitiveType::Float16 => new_full_null!(pf16),
        PrimitiveType::Float32 => new_full_null!(f32),
        PrimitiveType::Float64 => new_full_null!(f64),
        PrimitiveType::DaysMs => new_full_null!(days_ms),
        PrimitiveType::MonthDayNano => new_full_null!(months_days_ns),
        PrimitiveType::MonthDayMillis => unimplemented!(
            "cannot build an array of months_days_ms elements: they are of no rust type an array \
             of polars-array is taken over",
        ),
    }
}

/// Panics unless the `length` elements at `start` are in bounds of `array_len` elements.
pub(crate) fn assert_subslice(array_len: usize, start: usize, length: usize) {
    assert!(
        start
            .checked_add(length)
            .is_some_and(|end| end <= array_len),
        "subslice of {length} elements at {start} is out of bounds of an array of length \
         {array_len}",
    );
}

/// Appends the `length` bits of `validity` starting at `start` to `dst`.
#[inline(never)]
pub fn subslice_extend_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    start: usize,
    length: usize,
) {
    if length == 0 {
        return;
    }

    match validity {
        None => dst.extend_constant(length, true),
        Some(validity) => match validity.scalar_value() {
            Some(bit) => dst.extend_constant(length, bit),
            // The mask is not scalar, so it holds one bit per element.
            None => dst.subslice_extend_from_opt_validity(validity.flat_bitmap(), start, length),
        },
    }
}

/// Appends each of the `length` bits of `validity` starting at `start` `repeats` times over.
#[inline(never)]
pub fn subslice_extend_each_repeated_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    start: usize,
    length: usize,
    repeats: usize,
) {
    if length == 0 || repeats == 0 {
        return;
    }

    match validity {
        None => dst.extend_constant(length * repeats, true),
        Some(validity) => match validity.scalar_value() {
            Some(bit) => dst.extend_constant(length * repeats, bit),
            // The mask is not scalar, so it holds one bit per element.
            None => dst.subslice_extend_each_repeated_from_opt_validity(
                validity.flat_bitmap(),
                start,
                length,
                repeats,
            ),
        },
    }
}

/// Appends the bit of `validity` at every index of `idxs`, in the order they are given.
///
/// # Safety
/// Every index must be smaller than the length of `validity`.
#[inline(never)]
pub unsafe fn gather_extend_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    idxs: &[IdxSize],
) {
    match validity {
        None => dst.extend_constant(idxs.len(), true),
        Some(validity) => match validity.scalar_value() {
            Some(bit) => dst.extend_constant(idxs.len(), bit),
            // SAFETY: the mask is flat, so the indices are in bounds of the bitmap itself.
            None => unsafe {
                dst.gather_extend_from_opt_validity(validity.flat_bitmap(), idxs);
            },
        },
    }
}

/// Appends the bit of `validity` at every index of `idxs`; an index past `length` is unset.
#[inline(never)]
pub fn opt_gather_extend_validity(
    dst: &mut OptBitmapBuilder,
    validity: Option<PlBitmapRef<'_>>,
    idxs: &[IdxSize],
    length: usize,
) {
    match validity {
        None => dst.opt_gather_extend_from_opt_validity(None, idxs, length),
        Some(validity) => match validity.scalar_value() {
            // Every in-bounds index reads the one bit, and every other one is null.
            Some(bit) => {
                for idx in idxs {
                    dst.extend_constant(1, bit && (*idx as usize) < length);
                }
            },
            // The mask is not scalar, so it holds one bit per element.
            None => dst.opt_gather_extend_from_opt_validity(validity.flat_bitmap(), idxs, length),
        },
    }
}
