use polars_utils::IdxSize;

use crate::array::binary::BinaryArrayBuilder;
use crate::array::binview::BinaryViewArrayGenericBuilder;
use crate::array::boolean::BooleanArrayBuilder;
use crate::array::fixed_size_binary::FixedSizeBinaryArrayBuilder;
use crate::array::fixed_size_list::FixedSizeListArrayBuilder;
use crate::array::list::ListArrayBuilder;
use crate::array::null::NullArrayBuilder;
use crate::array::struct_::StructArrayBuilder;
use crate::array::{Array, PrimitiveArrayBuilder};
use crate::datatypes::{ArrowDataType, PhysicalType};
use crate::with_match_primitive_type_full;

/// Used for arrays which can share buffers with input arrays to appends,
/// gathers, etc.
#[derive(Copy, Clone, Debug)]
pub enum ShareStrategy {
    Never,
    Always,
}

pub trait StaticArrayBuilder: Send {
    type Array: Array;

    fn dtype(&self) -> &ArrowDataType;
    fn reserve(&mut self, additional: usize);

    /// Consume this builder returning the built array.
    fn freeze(self) -> Self::Array;

    /// Return the built array and reset to an empty state.
    fn freeze_reset(&mut self) -> Self::Array;

    /// Returns the length of this builder (so far).
    fn len(&self) -> usize;

    /// Extend this builder with the given number of null elements.
    fn extend_nulls(&mut self, length: usize);

    /// Extends this builder with the contents of the given array. May panic if
    /// other does not match the dtype of this array.
    fn extend(&mut self, other: &Self::Array, share: ShareStrategy) {
        self.subslice_extend(other, 0, other.len(), share);
    }

    /// Extends this builder with the contents of the given array subslice. May
    /// panic if other does not match the dtype of this array.
    fn subslice_extend(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    );

    /// The same as subslice_extend, but repeats the extension `repeats` times.
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
            self.subslice_extend(other, start, length, share)
        }
    }

    /// The same as subslice_extend, but repeats each element `repeats` times.
    fn subslice_extend_each_repeated(
        &mut self,
        other: &Self::Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Extends this builder with the contents of the given array at the given
    /// indices. That is, `other[idxs[i]]` is appended to this array in order,
    /// for each i=0..idxs.len(). May panic if other does not match the
    /// dtype of this array.
    ///
    /// # Safety
    /// The indices must be in-bounds.
    unsafe fn gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy);

    /// Extends this builder with the contents of the given array at the given
    /// indices. That is, `other[idxs[i]]` is appended to this array in order,
    /// for each i=0..idxs.len(). May panic if other does not match the
    /// dtype of this array. Out-of-bounds indices are mapped to nulls.
    fn opt_gather_extend(&mut self, other: &Self::Array, idxs: &[IdxSize], share: ShareStrategy);
}

impl<T: StaticArrayBuilder> ArrayBuilder for T {
    #[inline(always)]
    fn dtype(&self) -> &ArrowDataType {
        StaticArrayBuilder::dtype(self)
    }

    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        StaticArrayBuilder::reserve(self, additional)
    }

    #[inline(always)]
    fn freeze(self) -> Box<dyn Array> {
        Box::new(StaticArrayBuilder::freeze(self))
    }

    #[inline(always)]
    fn freeze_reset(&mut self) -> Box<dyn Array> {
        Box::new(StaticArrayBuilder::freeze_reset(self))
    }

    #[inline(always)]
    fn len(&self) -> usize {
        StaticArrayBuilder::len(self)
    }

    #[inline(always)]
    fn extend_nulls(&mut self, length: usize) {
        StaticArrayBuilder::extend_nulls(self, length);
    }

    #[inline(always)]
    fn subslice_extend(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        let other: &T::Array = other.as_any().downcast_ref().unwrap();
        StaticArrayBuilder::subslice_extend(self, other, start, length, share);
    }

    #[inline(always)]
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let other: &T::Array = other.as_any().downcast_ref().unwrap();
        StaticArrayBuilder::subslice_extend_repeated(self, other, start, length, repeats, share);
    }

    #[inline(always)]
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        let other: &T::Array = other.as_any().downcast_ref().unwrap();
        StaticArrayBuilder::subslice_extend_each_repeated(
            self, other, start, length, repeats, share,
        );
    }

    #[inline(always)]
    unsafe fn gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy) {
        let other: &T::Array = other.as_any().downcast_ref().unwrap();
        StaticArrayBuilder::gather_extend(self, other, idxs, share);
    }

    #[inline(always)]
    fn opt_gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy) {
        let other: &T::Array = other.as_any().downcast_ref().unwrap();
        StaticArrayBuilder::opt_gather_extend(self, other, idxs, share);
    }
}

#[allow(private_bounds)]
pub trait ArrayBuilder: ArrayBuilderBoxedHelper + Send {
    fn dtype(&self) -> &ArrowDataType;
    fn reserve(&mut self, additional: usize);

    /// Consume this builder returning the built array.
    fn freeze(self) -> Box<dyn Array>;

    /// Return the built array and reset to an empty state.
    fn freeze_reset(&mut self) -> Box<dyn Array>;

    /// Returns the length of this builder (so far).
    fn len(&self) -> usize;

    /// Extend this builder with the given number of null elements.
    fn extend_nulls(&mut self, length: usize);

    /// Extends this builder with the contents of the given array. May panic if
    /// other does not match the dtype of this array.
    fn extend(&mut self, other: &dyn Array, share: ShareStrategy) {
        self.subslice_extend(other, 0, other.len(), share);
    }

    /// Extends this builder with the contents of the given array subslice. May
    /// panic if other does not match the dtype of this array.
    fn subslice_extend(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    );

    /// The same as subslice_extend, but repeats the extension `repeats` times.
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// The same as subslice_extend, but repeats each element `repeats` times.
    fn subslice_extend_each_repeated(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    );

    /// Extends this builder with the contents of the given array at the given
    /// indices. That is, `other[idxs[i]]` is appended to this array in order,
    /// for each i=0..idxs.len(). May panic if other does not match the
    /// dtype of this array.
    ///
    /// # Safety
    /// The indices must be in-bounds.
    unsafe fn gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy);

    /// Extends this builder with the contents of the given array at the given
    /// indices. That is, `other[idxs[i]]` is appended to this array in order,
    /// for each i=0..idxs.len(). May panic if other does not match the
    /// dtype of this array. Out-of-bounds indices are mapped to nulls.
    fn opt_gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy);
}

/// A hack that lets us call the consuming `freeze` method on Box<dyn ArrayBuilder>.
trait ArrayBuilderBoxedHelper {
    fn freeze_boxed(self: Box<Self>) -> Box<dyn Array>;
}

impl<T: ArrayBuilder> ArrayBuilderBoxedHelper for T {
    fn freeze_boxed(self: Box<Self>) -> Box<dyn Array> {
        self.freeze()
    }
}

impl ArrayBuilder for Box<dyn ArrayBuilder> {
    #[inline(always)]
    fn dtype(&self) -> &ArrowDataType {
        (**self).dtype()
    }

    #[inline(always)]
    fn reserve(&mut self, additional: usize) {
        (**self).reserve(additional)
    }

    #[inline(always)]
    fn freeze(self) -> Box<dyn Array> {
        self.freeze_boxed()
    }

    #[inline(always)]
    fn freeze_reset(&mut self) -> Box<dyn Array> {
        (**self).freeze_reset()
    }

    #[inline(always)]
    fn len(&self) -> usize {
        (**self).len()
    }

    #[inline(always)]
    fn extend_nulls(&mut self, length: usize) {
        (**self).extend_nulls(length);
    }

    #[inline(always)]
    fn subslice_extend(
        &mut self,
        other: &dyn Array,
        start: usize,
        length: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend(other, start, length, share);
    }

    #[inline(always)]
    fn subslice_extend_repeated(
        &mut self,
        other: &dyn Array,
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
        other: &dyn Array,
        start: usize,
        length: usize,
        repeats: usize,
        share: ShareStrategy,
    ) {
        (**self).subslice_extend_each_repeated(other, start, length, repeats, share);
    }

    #[inline(always)]
    unsafe fn gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy) {
        (**self).gather_extend(other, idxs, share);
    }

    #[inline(always)]
    fn opt_gather_extend(&mut self, other: &dyn Array, idxs: &[IdxSize], share: ShareStrategy) {
        (**self).opt_gather_extend(other, idxs, share);
    }
}

/// Construct an ArrayBuilder for the given type.
pub fn make_builder(dtype: &ArrowDataType) -> Box<dyn ArrayBuilder> {
    use PhysicalType::*;
    match dtype.to_physical_type() {
        Null => Box::new(NullArrayBuilder::new(dtype.clone())),
        Boolean => Box::new(BooleanArrayBuilder::new(dtype.clone())),
        Primitive(prim_t) => with_match_primitive_type_full!(prim_t, |$T| {
            Box::new(PrimitiveArrayBuilder::<$T>::new(dtype.clone()))
        }),
        LargeBinary => Box::new(BinaryArrayBuilder::<i64>::new(dtype.clone())),
        FixedSizeBinary => Box::new(FixedSizeBinaryArrayBuilder::new(dtype.clone())),
        LargeList => {
            let ArrowDataType::LargeList(inner_dt) = dtype else {
                unreachable!()
            };
            Box::new(ListArrayBuilder::<i64, _>::new(
                dtype.clone(),
                make_builder(inner_dt.dtype()),
            ))
        },
        FixedSizeList => {
            let ArrowDataType::FixedSizeList(inner_dt, _) = dtype else {
                unreachable!()
            };
            Box::new(FixedSizeListArrayBuilder::new(
                dtype.clone(),
                make_builder(inner_dt.dtype()),
            ))
        },
        Struct => {
            let ArrowDataType::Struct(fields) = dtype else {
                unreachable!()
            };
            let builders = fields.iter().map(|f| make_builder(f.dtype())).collect();
            Box::new(StructArrayBuilder::new(dtype.clone(), builders))
        },
        BinaryView => Box::new(BinaryViewArrayGenericBuilder::<[u8]>::new(dtype.clone())),
        Utf8View => Box::new(BinaryViewArrayGenericBuilder::<str>::new(dtype.clone())),

        List | Binary | Utf8 | LargeUtf8 | Map | Union | Dictionary(_) => {
            unimplemented!()
        },
    }
}
