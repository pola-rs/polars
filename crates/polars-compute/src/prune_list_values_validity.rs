use arrow::array::{Array, FixedSizeListArray, ListArray};
use arrow::bitmap::bitmask::BitMask;
use arrow::types::Offset;

/// Removes validity mask from list values if all bits that fall within the
/// offsets are set.
pub fn prune_list_values_validity<O: Offset>(arr: &ListArray<O>) -> Option<ListArray<O>> {
    let values = arr.values();

    let values_validity = values.validity()?;

    let list_validity = arr.validity();

    let list_validity = list_validity.map(BitMask::from_bitmap);
    let values_validity = BitMask::from_bitmap(values_validity);

    if values_validity.unset_bits() > 0 {
        let offsets = arr.offsets();

        let mut has_unset = false;

        assert!(list_validity.is_none_or(|x| x.len() == offsets.len_proxy()));
        assert_eq!(values_validity.len(), offsets.last().to_usize());

        for i in 0..offsets.len_proxy() {
            let (start, end) = offsets.start_end(i);

            has_unset |= list_validity.is_none_or(|x| unsafe { x.get_bit_unchecked(i) })
                && unsafe { values_validity.sliced_unchecked(start, end - start) }.unset_bits() > 0;
        }

        if has_unset {
            return None;
        }
    }

    Some(ListArray::new(
        arr.dtype().clone(),
        arr.offsets().clone(),
        values.with_validity(None),
        arr.validity().cloned(),
    ))
}

#[cfg(feature = "dtype-array")]
pub fn prune_fixed_size_list_values_validity(
    arr: &FixedSizeListArray,
) -> Option<FixedSizeListArray> {
    let values = arr.values();

    let width = arr.size();

    if width > 0
        && let Some(values_validity) = values.validity()
    {
        let list_validity = arr.validity().filter(|x| x.unset_bits() > 0)?;

        let mut has_unset = false;
        let values_validity = BitMask::from_bitmap(values_validity);

        assert_eq!(list_validity.len(), values_validity.len());

        for i in list_validity.true_idx_iter() {
            has_unset |= values_validity.sliced(i * width, width).unset_bits() > 0;
        }

        if has_unset {
            return None;
        }
    }

    Some(FixedSizeListArray::new(
        arr.dtype().clone(),
        arr.len(),
        values.with_validity(None),
        arr.validity().cloned(),
    ))
}
