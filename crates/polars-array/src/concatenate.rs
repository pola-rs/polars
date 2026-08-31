//! Concatenation of arrays into a single array of the same physical representation.
//!
//! [`concatenate`] appends arrays of the same [`PlArrayType`] in order; the typed functions next to
//! it do the same for one concrete array type, without going through a trait object. There is no
//! logical type to agree on here — that lives at a higher level — so what these functions require
//! is that the physical representations match: a boolean array does not concatenate with a
//! primitive one, and primitive arrays of different element types do not concatenate with each
//! other.
//!
//! # The scalar representation
//!
//! Concatenation is `O(total length)` in general: the elements of the result come from more than
//! one array, so a buffer that is scalar in an input has to be materialized. The cases where it
//! does not are the ones that matter for arrays whose length is unbounded by their memory use, and
//! they are kept `O(1)`:
//!
//! * Only one array holds elements: the result is that array, cloned.
//! * Every array stands for the same repeated element: the result is that element, repeated over
//!   the total length.
//! * Every element of every array is null: the result is a fully null array of the total length,
//!   whose values are undetermined and therefore need not be written out.
//!
//! A [`PlNullArray`] is nothing but a length, so concatenating null arrays is always `O(1)`. None
//! of these paths reaches the values of a [`PlListArray`] or the fields of a [`PlStructArray`], so
//! neither is checked for concatenability when one of them is taken.
//!
//! Outside those cases the values of the result are flat, but its validity mask still is not
//! materialized unless it has to be — see [`concatenate_validities`].

use arrow::bitmap::{Bitmap, BitmapBuilder};
use arrow::types::NativeType;
use polars_buffer::Buffer;
use polars_error::{PolarsResult, polars_bail, polars_ensure, polars_err};

use crate::array::PlArray;
use crate::array_type::PlArrayType;
use crate::{
    PlBooleanArray, PlListArray, PlNullArray, PlPrimitiveArray, PlStructArray,
    with_match_pl_primitive_array_type,
};

/// Concatenates `arrays`, in order, into a single array of their common [`PlArrayType`].
///
/// This dispatches to the typed function for that array type; see the [module docs](self) for when
/// the result keeps the scalar representation instead of being materialized.
///
/// Only the array types of the arrays themselves are checked, since that is all a
/// [`PlArrayType`] carries: what a nested array is taken over is checked where it is concatenated,
/// which is not reached when a fast path hands back one of the arrays unchanged.
///
/// # Errors
/// This function errors if `arrays` is empty, if the arrays do not all have the same
/// [`PlArrayType`], if primitive arrays of that type do not all have the same element type, or if
/// the values of nested arrays do not concatenate.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate;
/// use polars_array::{PlArray, PlPrimitiveArray};
///
/// let lhs = PlPrimitiveArray::from_vec(vec![1i32, 2]);
/// let rhs = PlPrimitiveArray::from_iter([Some(3i32), None]);
/// let concatenated = concatenate(&[&lhs, &rhs]).unwrap();
///
/// assert_eq!(concatenated.len(), 4);
/// assert_eq!(concatenated.null_count(), 1);
/// assert_eq!(
///     concatenated
///         .as_any()
///         .downcast_ref::<PlPrimitiveArray<i32>>()
///         .unwrap()
///         .iter()
///         .collect::<Vec<_>>(),
///     [Some(1), Some(2), Some(3), None],
/// );
/// ```
pub fn concatenate(arrays: &[&dyn PlArray]) -> PolarsResult<Box<dyn PlArray>> {
    let Some((first, rest)) = arrays.split_first() else {
        polars_bail!(InvalidOperation: "cannot concatenate an empty list of arrays");
    };

    let array_type = first.array_type();
    for array in rest {
        polars_ensure!(
            array.array_type() == array_type,
            InvalidOperation:
            "cannot concatenate an array of type {:?} with one of type {:?}",
            array_type, array.array_type(),
        );
    }

    match array_type {
        PlArrayType::Boolean => Ok(Box::new(concatenate_boolean(&downcast_all(arrays)?))),
        PlArrayType::Primitive(primitive) => {
            with_match_pl_primitive_array_type!(*first, |T| concatenate_primitive_as::<T>(arrays))
                .flatten()
                .ok_or_else(|| {
                    polars_err!(
                        InvalidOperation:
                        "cannot concatenate arrays of primitive type {:?} that do not all have \
                         the same element type",
                        primitive,
                    )
                })
        },
        PlArrayType::Struct => Ok(Box::new(concatenate_struct(&downcast_all(arrays)?)?)),
        PlArrayType::List => Ok(Box::new(concatenate_list(&downcast_all(arrays)?)?)),
        PlArrayType::Null => Ok(Box::new(concatenate_null(&downcast_all(arrays)?))),
    }
}

/// Concatenates the validity masks of `arrays`, in order, into the mask of their concatenation.
///
/// Returns `None` when no element of any array is null, which is the mask of a fully valid array,
/// and a scalar (one-bit) mask when every element of every array is null. Otherwise the mask is
/// materialized, one bit per element, which is `O(total length)`.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_validities;
/// use polars_array::PlPrimitiveArray;
///
/// // Nothing is null, so there is no mask to build.
/// let valid = PlPrimitiveArray::from_vec(vec![1i32, 2]);
/// assert!(concatenate_validities(&[&valid, &valid]).is_none());
///
/// // Everything is null, so a single bit stands for a billion elements.
/// let null = PlPrimitiveArray::<i32>::new_full_null(1_000_000_000);
/// let validity = concatenate_validities(&[&null, &null]).unwrap();
/// assert_eq!(validity.len(), 1);
/// ```
pub fn concatenate_validities<A: PlArray + ?Sized>(arrays: &[&A]) -> Option<Bitmap> {
    let (length, null_count) = total_length_and_null_count(arrays);
    concatenate_validities_with(arrays, length, null_count)
}

/// [`concatenate_validities`], for a caller that has already counted the elements and the nulls.
fn concatenate_validities_with<A: PlArray + ?Sized>(
    arrays: &[&A],
    length: usize,
    null_count: usize,
) -> Option<Bitmap> {
    if null_count == 0 {
        return None;
    }
    if null_count == length {
        // Every element is null, so a single shared bit is the whole mask.
        return Some(Bitmap::new_zeroed(1));
    }

    let mut validity = BitmapBuilder::with_capacity(length);
    for array in arrays {
        let null_count = array.null_count();
        if null_count == 0 {
            validity.extend_constant(array.len(), true);
        } else if null_count == array.len() {
            validity.extend_constant(array.len(), false);
        } else {
            // The mask is neither all-set nor all-unset, so it cannot be scalar: this is a clone.
            validity.extend_from_bitmap(&array.validity().unwrap().to_flat());
        }
    }
    validity.into_opt_validity()
}

/// Concatenates `arrays`, in order, into a single [`PlPrimitiveArray`].
///
/// See the [module docs](self) for when the result keeps the scalar representation instead of
/// being materialized. Concatenating no arrays yields an empty array.
///
/// # Example
/// ```
/// use polars_array::concatenate::concatenate_primitive;
/// use polars_array::PlPrimitiveArray;
///
/// // Two arrays standing for the same repeated element concatenate in `O(1)`.
/// let lhs = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
/// let rhs = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
/// let concatenated = concatenate_primitive(&[&lhs, &rhs]);
///
/// assert_eq!(concatenated.len(), 2_000_000_000);
/// assert_eq!(concatenated.values().len(), 1);
/// assert!(concatenated.is_scalar());
/// ```
pub fn concatenate_primitive<T: NativeType>(
    arrays: &[&PlPrimitiveArray<T>],
) -> PlPrimitiveArray<T> {
    if let Some(array) = only_non_empty(arrays) {
        return array.clone();
    }

    let (length, null_count) = total_length_and_null_count(arrays);

    // Every element is null, so every value is undetermined and none of them has to be written.
    if length > 0 && null_count == length {
        return PlPrimitiveArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(arrays, PlPrimitiveArray::scalar_value) {
        return match element {
            Some(value) => PlPrimitiveArray::new_scalar(value, length),
            None => PlPrimitiveArray::new_full_null(length),
        };
    }

    let validity = concatenate_validities_with(arrays, length, null_count);

    let mut values = Vec::with_capacity(length);
    for array in arrays.iter().filter(|array| !array.is_empty()) {
        if array.values_are_scalar() {
            values.extend(std::iter::repeat_n(array.value(0), array.len()));
        } else {
            values.extend_from_slice(array.values().as_slice());
        }
    }

    // SAFETY: the values hold one slot per element of the concatenation, and the mask is the one
    // `concatenate_validities_with` built for that many elements.
    unsafe { PlPrimitiveArray::new_unchecked(Buffer::from(values), length, validity) }
}

/// Concatenates `arrays`, in order, into a single [`PlBooleanArray`].
///
/// See the [module docs](self) for when the result keeps the scalar representation instead of
/// being materialized. Concatenating no arrays yields an empty array.
pub fn concatenate_boolean(arrays: &[&PlBooleanArray]) -> PlBooleanArray {
    if let Some(array) = only_non_empty(arrays) {
        return array.clone();
    }

    let (length, null_count) = total_length_and_null_count(arrays);

    // Every element is null, so every value is undetermined and none of them has to be written.
    if length > 0 && null_count == length {
        return PlBooleanArray::new_full_null(length);
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(arrays, PlBooleanArray::scalar_value) {
        return match element {
            Some(value) => PlBooleanArray::new_scalar(value, length),
            None => PlBooleanArray::new_full_null(length),
        };
    }

    let validity = concatenate_validities_with(arrays, length, null_count);

    let mut values = BitmapBuilder::with_capacity(length);
    for array in arrays.iter().filter(|array| !array.is_empty()) {
        let array_values = array.values();
        if array_values.is_scalar() {
            values.extend_constant(array.len(), array_values.get(0));
        } else {
            values.extend_from_bitmap(array_values.bitmap());
        }
    }

    // SAFETY: the values hold one bit per element of the concatenation, and the mask is the one
    // `concatenate_validities_with` built for that many elements.
    unsafe { PlBooleanArray::new_unchecked(values.freeze(), length, validity) }
}

/// Concatenates `arrays`, in order, into a single [`PlNullArray`].
///
/// A null array is nothing but a length, so this is `O(1)`: the lengths are added up and every
/// element of the result is null, like every element of every input. Concatenating no arrays
/// yields an empty array.
pub fn concatenate_null(arrays: &[&PlNullArray]) -> PlNullArray {
    PlNullArray::new(total_length_and_null_count(arrays).0)
}

/// Concatenates `arrays`, in order, into a single [`PlStructArray`], concatenating each field with
/// the field at the same position of every other array.
///
/// The fields carry their own representation, so a struct array of nothing but scalar fields
/// concatenates in `O(1)` exactly when its fields do. Concatenating no arrays yields an empty
/// array without fields.
///
/// # Errors
/// This function errors if the arrays do not all have the same number of fields, or if any of the
/// fields do not concatenate.
pub fn concatenate_struct(arrays: &[&PlStructArray]) -> PolarsResult<PlStructArray> {
    let Some((first, rest)) = arrays.split_first() else {
        return Ok(PlStructArray::new_empty());
    };

    let num_fields = first.num_fields();
    for array in rest {
        polars_ensure!(
            array.num_fields() == num_fields,
            InvalidOperation:
            "cannot concatenate a struct array of {} fields with one of {} fields",
            num_fields, array.num_fields(),
        );
    }

    if let Some(array) = only_non_empty(arrays) {
        return Ok(array.clone());
    }

    let (length, null_count) = total_length_and_null_count(arrays);

    let fields = (0..num_fields)
        .map(|i| {
            let fields = arrays
                .iter()
                .map(|array| array.field(i))
                .collect::<Vec<_>>();
            concatenate(&fields)
        })
        .collect::<PolarsResult<Vec<_>>>()?;

    let validity = concatenate_validities_with(arrays, length, null_count);

    // SAFETY: every field is the concatenation of the fields at that position, so it holds as many
    // elements as the arrays together, and the mask is the one `concatenate_validities_with` built
    // for that many elements.
    Ok(unsafe { PlStructArray::new_unchecked(fields, length, validity) })
}

/// Concatenates `arrays`, in order, into a single [`PlListArray`] over the concatenation of the
/// values their lists reach.
///
/// The offsets are rebased onto that values array, and the values an input holds outside its own
/// offsets — which slicing leaves behind — are dropped. See the [module docs](self) for when the
/// result keeps the scalar representation; outside those cases the offsets of the result are flat,
/// so an input whose own offsets are scalar has its element written out once per element it stands
/// for, since no two elements of a flat list array can cover the same range.
///
/// # Errors
/// This function errors if `arrays` is empty, since there is no values array to take the lists of
/// the result over, or if the values do not concatenate.
pub fn concatenate_list(arrays: &[&PlListArray]) -> PolarsResult<PlListArray> {
    if arrays.is_empty() {
        polars_bail!(
            InvalidOperation:
            "cannot concatenate an empty list of list arrays: there is no values array to take \
             the lists of the result over"
        );
    }

    if let Some(array) = only_non_empty(arrays) {
        return Ok(array.clone());
    }

    let (length, null_count) = total_length_and_null_count(arrays);

    // Every element is null, so every list is undetermined and none of them has to be written out;
    // the values array is what determines the type of the lists, of which the first one will do.
    if length > 0 && null_count == length {
        return Ok(PlListArray::new_full_null(
            arrays[0].values().to_boxed(),
            length,
        ));
    }

    // Every array stands for the same repeated element, which the result repeats over all of them.
    if let Some(element) = shared_element(arrays, PlListArray::scalar_value) {
        return Ok(match element {
            Some(element) => PlListArray::new_scalar(element, length),
            None => PlListArray::new_full_null(arrays[0].values().to_boxed(), length),
        });
    }

    // The values of each array are sliced to what its offsets reach, so that the offsets of the
    // result can be rebased onto their concatenation.
    let mut values = Vec::with_capacity(arrays.len());
    let mut offsets = Vec::with_capacity(length + 1);
    offsets.push(0);

    let mut end = 0;
    for array in arrays {
        let array_offsets = array.offsets();
        let first = array_offsets[0];

        if array.is_empty() {
            // No element of the array is reachable, but its values are still what the type of its
            // lists is taken from, which every array has to agree on.
            values.push(array.values().sliced(first as usize, 0));
        } else if array.offsets_are_scalar() {
            // Every element of the array covers the same range, which the result writes out once
            // per element: concatenating the element with copies of itself is what repeats it, and
            // that keeps the values scalar when the element is itself a single repeated value.
            let last = array_offsets[1];
            let element = array
                .values()
                .sliced(first as usize, (last - first) as usize);
            values.push(concatenate(&vec![&*element; array.len()])?);

            let value_length = last - first;
            offsets.extend((1..=array.len() as u64).map(|i| end + i * value_length));
            end += value_length * array.len() as u64;
        } else {
            let last = array_offsets[array.len()];
            values.push(
                array
                    .values()
                    .sliced(first as usize, (last - first) as usize),
            );
            offsets.extend(
                array_offsets[1..]
                    .iter()
                    .map(|offset| end + (offset - first)),
            );
            end += last - first;
        }
    }

    let values = concatenate(&values.iter().map(|values| &**values).collect::<Vec<_>>())?;
    let validity = concatenate_validities_with(arrays, length, null_count);

    // SAFETY: the offsets are the lengths of the lists of every array in order, so they are
    // ordered, there is one per element plus the end of the last, and the last of them is the
    // length of the values they were rebased onto. The mask is the one
    // `concatenate_validities_with` built for that many elements.
    Ok(unsafe { PlListArray::new_unchecked(values, Buffer::from(offsets), length, validity) })
}

/// The one array that holds every element of the concatenation, if the others are all empty.
///
/// Returns the first array when they are all empty: the concatenation is empty as well, and the
/// first array is what carries anything a [`PlArrayType`] does not — the values of a list array,
/// the fields of a struct array. Returns `None` when more than one array holds elements, and for
/// no arrays at all, which has no array to hand back.
fn only_non_empty<'a, A: PlArray + ?Sized>(arrays: &[&'a A]) -> Option<&'a A> {
    let mut non_empty = arrays.iter().copied().filter(|array| !array.is_empty());
    match non_empty.next() {
        Some(array) => non_empty.next().is_none().then_some(array),
        None => arrays.first().copied(),
    }
}

/// The total number of elements of `arrays`, and how many of them are null.
///
/// # Panics
/// Panics if the total length overflows a `usize`, which the scalar representation makes possible
/// without the memory to back it.
fn total_length_and_null_count<A: PlArray + ?Sized>(arrays: &[&A]) -> (usize, usize) {
    let mut length = 0usize;
    let mut null_count = 0usize;
    for array in arrays {
        length = length
            .checked_add(array.len())
            .expect("the total length of the concatenation overflows a `usize`");
        null_count += array.null_count();
    }
    (length, null_count)
}

/// The element every element of every array equals, if `element` sees one for each of them and
/// they all agree.
///
/// The arrays that hold no elements are skipped: they have no element to disagree. Returns `None`
/// when no array holds elements, since there is then no element to repeat.
fn shared_element<A: PlArray, T: PartialEq>(
    arrays: &[&A],
    element: impl Fn(&A) -> Option<T>,
) -> Option<T> {
    let mut shared = None;
    for array in arrays.iter().filter(|array| !array.is_empty()) {
        let element = element(array)?;
        match &shared {
            Some(shared) if *shared != element => return None,
            Some(_) => {},
            None => shared = Some(element),
        }
    }
    shared
}

/// Downcasts every array to `T`, which their equal [`PlArrayType`] guarantees is their concrete
/// array type — unless one of them is an outside implementation of [`PlArray`] reporting an array
/// type that is not its own.
fn downcast_all<'a, T: PlArray>(arrays: &[&'a dyn PlArray]) -> PolarsResult<Vec<&'a T>> {
    arrays
        .iter()
        .map(|array| {
            array.as_any().downcast_ref::<T>().ok_or_else(|| {
                polars_err!(
                    InvalidOperation:
                    "cannot concatenate arrays that report the array type {:?} but are not all of \
                     the same concrete array type",
                    array.array_type(),
                )
            })
        })
        .collect()
}

/// Concatenates `arrays` as [`PlPrimitiveArray<T>`], or returns `None` if that is not the concrete
/// array type of every one of them.
///
/// The element type this is called with is the one of the first array, which the others have to
/// agree on: their equal [`PlArrayType`] does not make them agree, since a
/// [`PrimitiveType`](crate::PrimitiveType) does not pin an element type down.
fn concatenate_primitive_as<T: NativeType>(arrays: &[&dyn PlArray]) -> Option<Box<dyn PlArray>> {
    let arrays = arrays
        .iter()
        .map(|array| array.as_any().downcast_ref::<PlPrimitiveArray<T>>())
        .collect::<Option<Vec<_>>>()?;
    Some(Box::new(concatenate_primitive(&arrays)))
}

#[cfg(test)]
mod tests {
    use arrow::array::View;
    use arrow::types::{days_ms, i256, months_days_ns};
    use polars_utils::float16::pf16;

    use super::*;

    /// The elements of a `PlPrimitiveArray<i32>`, whatever representation it is in.
    fn elements(array: &dyn PlArray) -> Vec<Option<i32>> {
        array
            .as_any()
            .downcast_ref::<PlPrimitiveArray<i32>>()
            .unwrap()
            .iter()
            .collect()
    }

    #[test]
    fn flat_arrays_are_appended_in_order() {
        let lhs = PlPrimitiveArray::from_vec(vec![1i32, 2]);
        let rhs = PlPrimitiveArray::from_iter([Some(3i32), None, Some(5)]);
        let concatenated = concatenate_primitive(&[&lhs, &rhs]);

        assert_eq!(concatenated.len(), 5);
        assert_eq!(concatenated.null_count(), 1);
        assert!(concatenated.is_flat());
        assert_eq!(
            concatenated.iter().collect::<Vec<_>>(),
            [Some(1), Some(2), Some(3), None, Some(5)],
        );
    }

    #[test]
    fn empty_arrays_contribute_nothing() {
        let empty = PlPrimitiveArray::<i32>::new_empty();
        let arr = PlPrimitiveArray::from_vec(vec![1i32, 2]);

        assert_eq!(concatenate_primitive(&[&empty, &arr, &empty]), arr);
        assert!(concatenate_primitive(&[&empty, &empty]).is_empty());

        // Concatenating no arrays at all is empty as well, and no more null than it is long.
        let concatenated = concatenate_primitive::<i32>(&[]);
        assert!(concatenated.is_empty());
        assert!(concatenated.validity().is_none());
        assert!(concatenate_boolean(&[]).validity().is_none());
    }

    #[test]
    fn the_only_array_with_elements_is_handed_back() {
        // A billion elements would not be walked in reasonable time; that this test finishes is
        // what shows the array is handed back rather than materialized.
        let scalar = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        let empty = PlPrimitiveArray::<i32>::new_empty();
        let concatenated = concatenate_primitive(&[&empty, &scalar, &empty]);

        assert_eq!(concatenated.len(), 1_000_000_000);
        assert_eq!(concatenated.values().len(), 1);
    }

    #[test]
    fn arrays_of_the_same_repeated_element_stay_scalar() {
        let scalar = PlPrimitiveArray::new_scalar(7i32, 1_000_000_000);
        // A flat array of length one stands for a single repeated element as well.
        let single = PlPrimitiveArray::from_vec(vec![7i32]);
        let concatenated = concatenate_primitive(&[&scalar, &single, &scalar]);

        assert_eq!(concatenated.len(), 2_000_000_001);
        assert!(concatenated.is_scalar());
        assert_eq!(concatenated.scalar_value(), Some(Some(7)));

        // A different element on either side leaves nothing to repeat.
        let sixes = PlPrimitiveArray::new_scalar(6i32, 3);
        let sevens = PlPrimitiveArray::new_scalar(7i32, 2);
        let concatenated = concatenate_primitive(&[&sixes, &sevens]);

        assert!(concatenated.is_flat());
        assert_eq!(concatenated.values().as_slice(), [6, 6, 6, 7, 7]);
    }

    #[test]
    fn arrays_of_nothing_but_nulls_stay_a_single_bit() {
        let scalar = PlPrimitiveArray::<i32>::new_full_null(1_000_000_000);
        let flat = PlPrimitiveArray::from_iter([None, None::<i32>]);
        let concatenated = concatenate_primitive(&[&scalar, &flat]);

        assert_eq!(concatenated.len(), 1_000_000_002);
        assert_eq!(concatenated.null_count(), 1_000_000_002);
        assert!(concatenated.validity().unwrap().is_scalar());
        // The values of null elements are undetermined, so none of them is written out.
        assert_eq!(concatenated.values().len(), 1);
    }

    #[test]
    fn a_mask_is_built_only_where_it_is_needed() {
        let valid = PlPrimitiveArray::from_vec(vec![1i32, 2]);
        let nulls = PlPrimitiveArray::<i32>::new_full_null(2);

        // No array has a null element, so the result carries no mask at all.
        assert!(
            concatenate_primitive(&[&valid, &valid])
                .validity()
                .is_none()
        );

        let concatenated = concatenate_primitive(&[&valid, &nulls, &valid]);
        assert!(!concatenated.validity().unwrap().is_scalar());
        assert_eq!(
            concatenated.iter().collect::<Vec<_>>(),
            [Some(1), Some(2), None, None, Some(1), Some(2)],
        );
    }

    #[test]
    fn masks_concatenate_on_their_own() {
        let valid = PlPrimitiveArray::from_vec(vec![1i32, 2]);
        let nulls = PlPrimitiveArray::<i32>::new_full_null(2);
        let some = PlPrimitiveArray::from_iter([Some(1i32), None]);

        assert!(concatenate_validities(&[&valid, &valid]).is_none());
        assert_eq!(concatenate_validities(&[&nulls, &nulls]).unwrap().len(), 1);
        assert_eq!(
            concatenate_validities(&[&valid, &nulls, &some]).unwrap(),
            Bitmap::from_iter([true, true, false, false, true, false]),
        );
    }

    #[test]
    fn boolean_arrays_are_appended() {
        let lhs = PlBooleanArray::from_vec(vec![true, false]);
        let rhs = PlBooleanArray::from_iter([Some(true), None]);
        let concatenated = concatenate_boolean(&[&lhs, &rhs]);

        assert_eq!(concatenated.len(), 4);
        assert_eq!(
            concatenated.iter().collect::<Vec<_>>(),
            [Some(true), Some(false), Some(true), None],
        );

        // A scalar array next to a flat one is materialized.
        let scalar = PlBooleanArray::new_scalar(true, 3);
        let concatenated = concatenate_boolean(&[&scalar, &lhs]);

        assert!(concatenated.is_flat());
        assert_eq!(
            concatenated.values().to_flat(),
            Bitmap::from_iter([true, true, true, true, false]),
        );

        // Two arrays of the same repeated element are not.
        let concatenated =
            concatenate_boolean(&[&scalar, &PlBooleanArray::new_scalar(true, 1_000_000_000)]);

        assert_eq!(concatenated.len(), 1_000_000_003);
        assert!(concatenated.is_scalar());
        assert_eq!(concatenated.scalar_value(), Some(Some(true)));

        // And neither are two of nothing but nulls.
        let nulls = PlBooleanArray::new_full_null(1_000_000_000);
        let concatenated = concatenate_boolean(&[&nulls, &nulls]);

        assert_eq!(concatenated.null_count(), 2_000_000_000);
        assert!(concatenated.validity().unwrap().is_scalar());
    }

    #[test]
    fn null_arrays_only_add_up_their_lengths() {
        let arr = PlNullArray::new(1_000_000_000);
        let concatenated = concatenate_null(&[&arr, &arr, &PlNullArray::new_empty()]);

        assert_eq!(concatenated.len(), 2_000_000_000);
        assert_eq!(concatenated.null_count(), 2_000_000_000);
        assert!(concatenate_null(&[]).is_empty());
    }

    #[test]
    fn struct_arrays_concatenate_their_fields() {
        let lhs = PlStructArray::from_fields(vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2])),
            Box::new(PlBooleanArray::from_vec(vec![true, false])),
        ]);
        let rhs = PlStructArray::new(
            vec![
                Box::new(PlPrimitiveArray::from_vec(vec![3i32])),
                Box::new(PlBooleanArray::from_vec(vec![true])),
            ],
            1,
            // The one row of this array is null.
            Some(Bitmap::new_zeroed(1)),
        );
        let concatenated = concatenate_struct(&[&lhs, &rhs]).unwrap();

        assert_eq!(concatenated.len(), 3);
        assert_eq!(concatenated.num_fields(), 2);
        assert_eq!(concatenated.null_count(), 1);
        assert!(concatenated.is_null(2));
        assert_eq!(elements(concatenated.field(0)), [Some(1), Some(2), Some(3)]);

        assert!(concatenate_struct(&[]).unwrap().is_empty());
    }

    #[test]
    fn struct_arrays_of_scalar_fields_stay_scalar() {
        let row = || {
            PlStructArray::from_fields(vec![
                Box::new(PlPrimitiveArray::<i32>::new_scalar(1, 1_000_000_000)),
                Box::new(PlBooleanArray::new_scalar(true, 1_000_000_000)),
            ])
        };
        let concatenated = concatenate_struct(&[&row(), &row()]).unwrap();

        assert_eq!(concatenated.len(), 2_000_000_000);
        assert!(concatenated.validity().is_none());
        for field in concatenated.fields() {
            assert_eq!(field.len(), 2_000_000_000);
        }
    }

    #[test]
    fn struct_arrays_of_different_fields_do_not_concatenate() {
        let lhs =
            PlStructArray::from_fields(vec![Box::new(PlPrimitiveArray::from_vec(vec![1i32]))]);
        let rhs = PlStructArray::from_fields(vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32])),
            Box::new(PlBooleanArray::from_vec(vec![true])),
        ]);
        let err = concatenate_struct(&[&lhs, &rhs]).unwrap_err().to_string();
        assert!(err.contains("1 fields with one of 2 fields"), "{err}");

        // The fields themselves have to concatenate as well.
        let boolean =
            PlStructArray::from_fields(vec![Box::new(PlBooleanArray::from_vec(vec![true]))]);
        assert!(concatenate_struct(&[&lhs, &boolean]).is_err());
    }

    #[test]
    fn list_arrays_rebase_their_offsets() {
        // The lists `[1, 2]`, `[]` and `[3, 4, 5]`.
        let lhs = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5])),
            Buffer::from(vec![0u64, 2, 2, 5]),
        );
        // The list `[7]` out of the middle of its values, and a null list.
        let rhs = PlListArray::new(
            Box::new(PlPrimitiveArray::from_vec(vec![6i32, 7, 8])),
            Buffer::from(vec![1u64, 2, 2]),
            2,
            Some(Bitmap::from_iter([true, false])),
        );
        let concatenated = concatenate_list(&[&lhs, &rhs]).unwrap();

        assert_eq!(concatenated.len(), 5);
        assert_eq!(concatenated.offsets().as_slice(), [0, 2, 2, 5, 6, 6]);
        assert_eq!(concatenated.null_count(), 1);
        assert!(concatenated.is_null(4));
        assert_eq!(elements(&*concatenated.value(3)), [Some(7)]);
        // Only the values the offsets reach are kept: `6` and `8` are left behind.
        assert_eq!(
            elements(concatenated.values()),
            [Some(1), Some(2), Some(3), Some(4), Some(5), Some(7)],
        );
    }

    #[test]
    fn sliced_list_arrays_keep_only_the_values_they_reach() {
        // The lists `[3, 4, 5]` and `[]`, over values that still hold `1` and `2` before them.
        let arr = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5])),
            Buffer::from(vec![0u64, 2, 2, 5, 5]),
        )
        .sliced(2, 2);
        let concatenated = concatenate_list(&[&arr, &arr]).unwrap();

        assert_eq!(concatenated.len(), 4);
        assert_eq!(concatenated.offsets().as_slice(), [0, 3, 3, 6, 6]);
        assert_eq!(concatenated.values().len(), 6);
        assert_eq!(
            elements(&*concatenated.value(2)),
            [Some(3), Some(4), Some(5)]
        );
    }

    #[test]
    fn list_arrays_standing_for_the_same_list_concatenate_into_that_list() {
        // The list `[3, 4, 5]`, over values that still hold `1` and `2` before it.
        let arr = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3, 4, 5])),
            Buffer::from(vec![0u64, 2, 2, 5]),
        )
        .sliced(2, 1);

        // Every array stands for the same one list, so the result stands for it as well: only the
        // values that list reaches are kept, and nothing is written out per element.
        let concatenated =
            concatenate_list(&[&arr, &arr.new_from_index(0, 1_000_000_000)]).unwrap();

        assert_eq!(concatenated.len(), 1_000_000_001);
        assert!(concatenated.is_scalar());
        assert_eq!(concatenated.offsets().as_slice(), [0, 3]);
        assert_eq!(concatenated.values().len(), 3);
        assert_eq!(
            elements(&*concatenated.value(1_000_000_000)),
            [Some(3), Some(4), Some(5)],
        );

        // Lists that do not agree are written out one per element instead.
        let other = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![9i32])),
            Buffer::from(vec![0u64, 1]),
        );
        let concatenated = concatenate_list(&[&arr.new_from_index(0, 2), &other]).unwrap();

        assert!(!concatenated.is_scalar());
        assert_eq!(concatenated.offsets().as_slice(), [0, 3, 6, 7]);
        assert_eq!(
            elements(concatenated.values()),
            [
                Some(3),
                Some(4),
                Some(5),
                Some(3),
                Some(4),
                Some(5),
                Some(9)
            ],
        );
    }

    #[test]
    fn fully_null_list_arrays_concatenate_without_writing_out_their_offsets() {
        let null = PlListArray::new_full_null(
            Box::new(PlPrimitiveArray::<i32>::new_empty()),
            1_000_000_000,
        );
        let concatenated = concatenate_list(&[&null, &null]).unwrap();

        assert_eq!(concatenated.len(), 2_000_000_000);
        assert_eq!(concatenated.null_count(), 2_000_000_000);
        assert!(concatenated.is_scalar());
        assert_eq!(concatenated.offsets().as_slice(), [0, 0]);
        assert_eq!(concatenated.validity().unwrap().bitmap().len(), 1);
    }

    #[test]
    fn a_list_array_of_scalar_offsets_is_written_out_once_per_element() {
        // A thousand copies of the list `[7; 1_000_000_000]`, in `O(1)` memory.
        let scalar = PlListArray::new_scalar(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 1_000_000_000)),
            1_000,
        );
        let flat = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::new_scalar(7i32, 1)),
            Buffer::from(vec![0u64, 1]),
        );

        // No two elements of a flat list array can cover the same range, so the offsets of the
        // result step through the repeated element — but its values stay scalar.
        let concatenated = concatenate_list(&[&scalar, &flat]).unwrap();

        assert_eq!(concatenated.len(), 1_001);
        assert!(!concatenated.is_scalar());
        assert_eq!(concatenated.offsets().len(), 1_002);
        assert_eq!(
            concatenated.value_range(999),
            999_000_000_000..1_000_000_000_000
        );
        assert_eq!(concatenated.value_length(1_000), 1);
        assert_eq!(concatenated.values().len(), 1_000_000_000_001);
        assert!(
            concatenated
                .values()
                .as_any()
                .downcast_ref::<PlPrimitiveArray<i32>>()
                .unwrap()
                .values_are_scalar()
        );
    }

    #[test]
    fn list_arrays_need_a_values_array_to_take_their_lists_over() {
        let err = concatenate_list(&[]).unwrap_err().to_string();
        assert!(err.contains("values array"), "{err}");

        // The values have to concatenate: lists of `i32` are not lists of `i64`.
        let offsets = Buffer::from(vec![0u64, 1]);
        let narrow = PlListArray::from_offsets(
            Box::new(PlPrimitiveArray::from_vec(vec![1i32])),
            offsets.clone(),
        );
        let wide =
            PlListArray::from_offsets(Box::new(PlPrimitiveArray::from_vec(vec![1i64])), offsets);
        assert!(concatenate_list(&[&narrow, &wide]).is_err());
    }

    #[test]
    fn nested_values_concatenate_recursively() {
        let list = |values: Vec<i32>| {
            let length = values.len() as u64;
            PlListArray::from_offsets(
                Box::new(PlStructArray::from_fields(vec![Box::new(
                    PlPrimitiveArray::from_vec(values),
                )])),
                Buffer::from(vec![0u64, length]),
            )
        };
        let concatenated = concatenate_list(&[&list(vec![1, 2]), &list(vec![3])]).unwrap();

        assert_eq!(concatenated.len(), 2);
        assert_eq!(concatenated.offsets().as_slice(), [0, 2, 3]);

        let values = concatenated
            .values()
            .as_any()
            .downcast_ref::<PlStructArray>()
            .unwrap();
        assert_eq!(values.len(), 3);
        assert_eq!(elements(values.field(0)), [Some(1), Some(2), Some(3)]);
    }

    #[test]
    fn the_trait_object_dispatches_to_every_array_type() {
        let arrays: Vec<Box<dyn PlArray>> = vec![
            Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
            Box::new(PlBooleanArray::from_vec(vec![true, false, true])),
            Box::new(PlNullArray::new(3)),
            Box::new(PlStructArray::from_fields(vec![Box::new(
                PlPrimitiveArray::from_vec(vec![1i32, 2, 3]),
            )])),
            Box::new(PlListArray::from_offsets(
                Box::new(PlPrimitiveArray::from_vec(vec![1i32, 2, 3])),
                Buffer::from(vec![0u64, 1, 2, 3]),
            )),
        ];

        for array in &arrays {
            let concatenated = concatenate(&[&**array, &**array]).unwrap();

            assert_eq!(concatenated.array_type(), array.array_type());
            assert_eq!(concatenated.len(), 6);
            assert_eq!(&concatenated.sliced(0, 3), array);
            assert_eq!(&concatenated.sliced(3, 3), array);
        }
    }

    #[test]
    fn every_element_type_a_primitive_array_can_hold_is_dispatched() {
        fn assert_dispatches<T: NativeType>() {
            let array = PlPrimitiveArray::from_vec(vec![T::default(), T::default()]);
            let concatenated = concatenate(&[&array, &array]).unwrap();

            assert_eq!(
                concatenated.array_type(),
                PlArrayType::Primitive(T::PRIMITIVE),
            );
            assert_eq!(concatenated.len(), 4);
            assert!(
                concatenated
                    .as_any()
                    .downcast_ref::<PlPrimitiveArray<T>>()
                    .is_some()
            );
        }

        assert_dispatches::<i8>();
        assert_dispatches::<i16>();
        assert_dispatches::<i32>();
        assert_dispatches::<i64>();
        assert_dispatches::<i128>();
        assert_dispatches::<i256>();
        assert_dispatches::<u8>();
        assert_dispatches::<u16>();
        assert_dispatches::<u32>();
        assert_dispatches::<u64>();
        assert_dispatches::<u128>();
        assert_dispatches::<pf16>();
        assert_dispatches::<f32>();
        assert_dispatches::<f64>();
        assert_dispatches::<days_ms>();
        assert_dispatches::<months_days_ns>();
        assert_dispatches::<View>();
    }

    #[test]
    fn arrays_of_different_types_do_not_concatenate() {
        let primitive: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i32]));
        let boolean: Box<dyn PlArray> = Box::new(PlBooleanArray::from_vec(vec![true]));
        let err = concatenate(&[&*primitive, &*boolean])
            .unwrap_err()
            .to_string();
        assert!(
            err.contains("Primitive(Int32)") && err.contains("Boolean"),
            "{err}"
        );

        // The element type is part of the array type: `i32` is not `i64`.
        let wide: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1i64]));
        assert!(concatenate(&[&*primitive, &*wide]).is_err());

        // `View` and `u128` share a primitive type without sharing an element type, so the arrays
        // of the two are told apart by the concrete type they downcast to.
        let views: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![View::default()]));
        let wide: Box<dyn PlArray> = Box::new(PlPrimitiveArray::from_vec(vec![1u128]));
        let err = concatenate(&[&*views, &*wide]).unwrap_err().to_string();
        assert!(err.contains("same element type"), "{err}");

        // There is nothing to concatenate without arrays.
        assert!(concatenate(&[]).is_err());
    }
}
