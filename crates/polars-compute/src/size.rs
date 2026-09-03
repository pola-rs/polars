use polars_array::{PlBinaryViewArray, PlPrimitiveArray};

/// The length in bytes of every element, read off the views.
pub fn binary_size_bytes(array: &PlBinaryViewArray) -> PlPrimitiveArray<u32> {
    // A scalar views buffer holds the one view every element reads: its length is measured once
    // and repeated in turn, in `O(1)` memory.
    let lengths = match array.scalar_views() {
        Some(view) => PlPrimitiveArray::new_scalar(view.length, array.len()),
        None => PlPrimitiveArray::from_vec(
            array
                .flat_views()
                .expect("a views buffer that is not scalar is flat")
                .iter()
                .map(|view| view.length)
                .collect(),
        ),
    };

    lengths.with_validity_broadcast(
        array
            .validity()
            .map(|validity| validity.to_flat_or_scalar()),
    )
}

#[cfg(test)]
mod tests {
    use arrow::bitmap::Bitmap;

    use super::*;

    /// The one view a scalar chunk holds is measured once, and the result repeats the length
    /// rather than holding a slot per element.
    #[test]
    fn a_repeated_value_keeps_its_length_repeated() {
        let scalar = PlBinaryViewArray::new_scalar(b"hello", 100);
        let sizes = binary_size_bytes(&scalar);

        assert!(sizes.values_are_scalar());
        assert_eq!(sizes, PlPrimitiveArray::from_vec(vec![5u32; 100]));
    }

    /// A chunk laid out one view per element is measured one element at a time, and the mask
    /// comes along as it is.
    #[test]
    fn every_element_is_measured() {
        let arr = PlBinaryViewArray::from_iter([Some(&b"a"[..]), None, Some(&b"three"[..])]);
        assert_eq!(
            binary_size_bytes(&arr),
            PlPrimitiveArray::from_iter([Some(1u32), None, Some(5)]),
        );

        // A scalar view under a mask that holds one bit per element stays scalar.
        let masked = PlBinaryViewArray::new_scalar(b"ab", 3)
            .with_validity_broadcast(Some(Bitmap::from_iter([true, false, true])));
        assert_eq!(
            binary_size_bytes(&masked),
            PlPrimitiveArray::from_iter([Some(2u32), None, Some(2)]),
        );
    }

    #[test]
    fn an_empty_chunk_measures_nothing() {
        let empty = PlBinaryViewArray::new_empty();
        assert!(binary_size_bytes(&empty).is_empty());
    }
}
