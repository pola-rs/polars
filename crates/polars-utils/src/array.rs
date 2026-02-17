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
