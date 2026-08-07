use std::mem::ManuallyDrop;

#[repr(C)]
struct ArrayPair<T, const NUM_LEFT: usize, const NUM_RIGHT: usize>([T; NUM_LEFT], [T; NUM_RIGHT]);

pub fn try_map<T, U, const N: usize>(
    array: [T; N],
    f: impl FnMut(T) -> Option<U>,
) -> Option<[U; N]> {
    let mut array = array.map(f);

    if array.iter().any(Option::is_none) {
        return None;
    }

    Some(std::array::from_fn(|n| array[n].take().unwrap()))
}

/// Concatenate 2 arrays.
pub fn array_concat<T, const NUM_LEFT: usize, const NUM_RIGHT: usize, const NUM_TOTAL: usize>(
    left: [T; NUM_LEFT],
    right: [T; NUM_RIGHT],
) -> [T; NUM_TOTAL] {
    const {
        assert!(NUM_LEFT + NUM_RIGHT == NUM_TOTAL);
    }

    unsafe { std::mem::transmute_copy(&ManuallyDrop::new(ArrayPair(left, right))) }
}

/// Split an array to 2 arrays.
pub fn array_split<T, const NUM_LEFT: usize, const NUM_RIGHT: usize, const NUM_TOTAL: usize>(
    array: [T; NUM_TOTAL],
) -> ([T; NUM_LEFT], [T; NUM_RIGHT]) {
    const {
        assert!(NUM_LEFT + NUM_RIGHT == NUM_TOTAL);
    }

    let ArrayPair::<T, NUM_LEFT, NUM_RIGHT>(l, r) =
        unsafe { std::mem::transmute_copy(&ManuallyDrop::new(array)) };

    (l, r)
}
