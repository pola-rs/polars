//! Contains the trait [`Growable`] and corresponding concreate implementations, one per concrete array,
//! that offer the ability to create a new [`Array`] out of slices of existing [`Array`]s.

use crate::array::*;
use crate::datatypes::*;

mod binary;
pub use binary::GrowableBinary;
mod boolean;
pub use boolean::GrowableBoolean;
mod fixed_binary;
pub use fixed_binary::GrowableFixedSizeBinary;
mod null;
pub use null::GrowableNull;
mod primitive;
pub use primitive::GrowablePrimitive;
mod list;
pub use list::GrowableList;
mod structure;
pub use structure::GrowableStruct;
mod fixed_size_list;
pub use fixed_size_list::GrowableFixedSizeList;
mod utf8;
pub use utf8::GrowableUtf8;
mod dictionary;
pub use dictionary::GrowableDictionary;

mod binview;
pub use binview::GrowableBinaryViewArray;

mod utils;

/// Describes a struct that can be extended from slices of other pre-existing [`Array`]s.
/// This is used in operations where a new array is built out of other arrays, such
/// as filter and concatenation.
pub trait Growable<'a> {
    /// Extends this [`Growable`] with elements from the bounded [`Array`] at index `index` from
    /// a slice starting at `start` and length `len`.
    ///
    /// # Safety
    /// Doesn't do any bound checks.
    unsafe fn extend(&mut self, index: usize, start: usize, len: usize);

    /// Same as extend, except it repeats the extension `copies` times.
    ///
    /// # Safety
    /// Doesn't do any bound checks.
    unsafe fn extend_copies(&mut self, index: usize, start: usize, len: usize, copies: usize) {
        for _ in 0..copies {
            self.extend(index, start, len)
        }
    }

    /// Extends this [`Growable`] with null elements, disregarding the bound arrays
    ///
    /// # Safety
    /// Doesn't do any bound checks
    fn extend_validity(&mut self, additional: usize);

    /// The current length of the [`Growable`].
    fn len(&self) -> usize;

    /// Converts this [`Growable`] to an [`Arc<dyn Array>`], thereby finishing the mutation.
    /// Self will be empty after such operation.
    fn as_arc(&mut self) -> Arc<dyn Array> {
        self.as_box().into()
    }

    /// Converts this [`Growable`] to an [`Box<dyn Array>`], thereby finishing the mutation.
    /// Self will be empty after such operation
    fn as_box(&mut self) -> Box<dyn Array>;
}

macro_rules! dyn_growable {
    ($ty:ty, $arrays:expr, $use_validity:expr, $capacity:expr) => {{
        let arrays = $arrays
            .iter()
            .map(|array| array.as_any().downcast_ref().unwrap())
            .collect::<Vec<_>>();
        Box::new(<$ty>::new(arrays, $use_validity, $capacity))
    }};
}

/// Creates a new [`Growable`] from an arbitrary number of [`Array`]s.
/// # Panics
/// This function panics iff
/// * the arrays do not have the same [`ArrowDataType`].
/// * `arrays.is_empty()`.
pub fn make_growable<'a>(
    arrays: &[&'a dyn Array],
    use_validity: bool,
    capacity: usize,
) -> Box<dyn Growable<'a> + 'a> {
    assert!(!arrays.is_empty());
    let data_type = arrays[0].data_type();

    use PhysicalType::*;
    match data_type.to_physical_type() {
        Null => Box::new(null::GrowableNull::new(data_type.clone())),
        Boolean => dyn_growable!(boolean::GrowableBoolean, arrays, use_validity, capacity),
        Primitive(primitive) => with_match_primitive_type_full!(primitive, |$T| {
            dyn_growable!(primitive::GrowablePrimitive::<$T>, arrays, use_validity, capacity)
        }),
        Binary => dyn_growable!(
            binary::GrowableBinary::<i32>,
            arrays,
            use_validity,
            capacity
        ),
        LargeBinary => dyn_growable!(
            binary::GrowableBinary::<i64>,
            arrays,
            use_validity,
            capacity
        ),
        FixedSizeBinary => dyn_growable!(
            fixed_binary::GrowableFixedSizeBinary,
            arrays,
            use_validity,
            capacity
        ),
        LargeList => dyn_growable!(list::GrowableList::<i64>, arrays, use_validity, capacity),
        Struct => dyn_growable!(structure::GrowableStruct, arrays, use_validity, capacity),
        FixedSizeList => dyn_growable!(
            fixed_size_list::GrowableFixedSizeList,
            arrays,
            use_validity,
            capacity
        ),
        BinaryView => {
            dyn_growable!(
                binview::GrowableBinaryViewArray::<[u8]>,
                arrays,
                use_validity,
                capacity
            )
        },
        Utf8View => {
            dyn_growable!(
                binview::GrowableBinaryViewArray::<str>,
                arrays,
                use_validity,
                capacity
            )
        },
        Dictionary(key_type) => {
            match_integer_type!(key_type, |$T| {
                let arrays = arrays
                    .iter()
                    .map(|array| {
                        array
                            .as_any()
                            .downcast_ref::<DictionaryArray<$T>>()
                            .unwrap()
                    })
                    .collect::<Vec<_>>();
                Box::new(dictionary::GrowableDictionary::<$T>::new(
                    &arrays,
                    use_validity,
                    capacity,
                ))
            })
        },
        Union | Map | Utf8 | LargeUtf8 | List => unimplemented!(),
    }
}
