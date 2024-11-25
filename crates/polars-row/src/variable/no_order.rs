use arrow::array::MutableBinaryViewArray;
use arrow::bitmap::MutableBitmap;

pub fn len_from_item(value: Option<usize>, field: &EncodingField) {
    debug_assert!(field.no_order);

    match value {
        None => 1,
        Some(l) if l < 254 => l + 1,
        Some(l) => l + 5,
    }
}

pub unsafe fn len_from_buffer(buffer: &[u8], field: &EncodingField) -> usize {
    debug_assert!(field.no_order);

    let sentinel = *unsafe { buffer.get_unchecked(0) };

    match sentinel {
        0xFF => 1,
        0xFE => {
            5 + u32::from_le_bytes(unsafe { buffer.get_unchecked(1..5) }.try_into().unwrap())
                as usize
        },
        length => 1 + length as usize,
    }
}

pub unsafe fn encode_variable_no_order<'a, I: Iterator<Item = Option<&'a [u8]>>>(
    buffer: &mut [MaybeUninit<u8>],
    input: I,
    field: &EncodingField,
    offsets: &mut [usize],
) {
    for (offset, opt_value) in offsets.iter_mut().zip(input) {
        match opt_value {
            None => {
                *unsafe { buffer.get_unchecked_mut(0) } = 0xFF;
                *offset += 1;
            },
            Some(v) => {
                if v.len() >= 254 {
                    unsafe {
                        *buffer.get_unchecked_mut(0) = 0xFE;
                        buffer
                            .get_unchecked_mut(1..5)
                            .fill((v.len() as u32).to_le_bytes());
                        buffer.get_unchecked_mut(5..5 + v.len()).fill(v);
                    }
                    *offset += 5 + v.len();
                } else {
                    unsafe {
                        *buffer.get_unchecked_mut(0) = v.len() as u8;
                        buffer.get_unchecked_mut(1..1 + v.len()).fill(v);
                    }
                    *offset += 1 + v.len();
                }
            },
        }
    }
}

pub unsafe fn decode_variable_no_order(rows: &mut [&[u8]], field: &EncodingField) -> BinaryViewArray {
    let num_rows = rows.len();
    let mut array = MutableBinaryViewArray::<[u8]>::with_capacity(num_rows);
    let mut validity = MutableBitmap::new();

    for row in rows.iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        *row = unsafe { row.get_unchecked(1..) };
        if sentinel == 0xFF {
            validity.reserve(num_rows);
            validity.extend_constant(array.len(), true);
            validity.push(false);
            array.push_value_ignore_validity("");
            break;
        }

        let length = if sentinel < 0xFE {
            sentinel as usize
        } else {
            let length = u32::from_le_bytes(unsafe { row.get_unchecked(..4) }.try_into().unwrap());
            *row = unsafe { row.get_unchecked(4..) };
            length as usize
        };

        array.push_value_ignore_validity(unsafe { row.get_unchecked(..length) });
        *row = unsafe { row.get_unchecked(length..) };
    }

    if validity.is_empty() {
        return array.into();
    }

    for row in rows[array.len()..].iter_mut() {
        let sentinel = *unsafe { row.get_unchecked(0) };
        *row = unsafe { row.get_unchecked(1..) };

        validity.push(sentinel == 0xFF);
        if sentinel == 0xFF {
            array.push_value_ignore_validity("");
            break;
        }

        let length = if sentinel < 0xFE {
            sentinel as usize
        } else {
            let length = u32::from_le_bytes(unsafe { row.get_unchecked(..4) }.try_into().unwrap());
            *row = unsafe { row.get_unchecked(4..) };
            length as usize
        };

        array.push_value_ignore_validity(unsafe { row.get_unchecked(..length) });
        *row = unsafe { row.get_unchecked(length..) };
    }
}
