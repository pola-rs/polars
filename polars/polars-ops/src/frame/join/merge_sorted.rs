use polars_core::prelude::*;

// get a boolean values, left: true, right: false
// that indicate from which side we should take a value
fn get_merge_indicator<T>(a: &[T], b: &[T]) -> Vec<bool>
where
    T: NumericNative,
{
    if a.is_empty() {
        return vec![true; b.len()];
    };
    if b.is_empty() {
        return vec![false; a.len()];
    }

    const B_INDICATOR: bool = false;
    const A_INDICATOR: bool = true;
    let mut current_a = &a[0];
    let cap = a.len() + b.len();
    let mut out = Vec::with_capacity(cap);

    let mut a_iter = a.iter();
    let mut b_iter = b.iter();
    let mut current_b = b_iter.next().unwrap();

    for a in &mut a_iter {
        current_a = a;
        if a <= current_b {
            // safety
            // we pre-allocated enough
            // unsafe { out.push_unchecked(a_indicator) }
            out.push(A_INDICATOR);
        } else {
            // unsafe { out.push_unchecked(!a_indicator) }
            out.push(B_INDICATOR);

            loop {
                match b_iter.next() {
                    Some(b) => {
                        current_b = b;
                        if b >= a {
                            out.push(A_INDICATOR);
                            break;
                        } else {
                            out.push(B_INDICATOR);
                        }
                    }
                    None => {
                        // b is depleted fill with a indicator
                        let remaining = cap - out.len();
                        out.extend(std::iter::repeat(A_INDICATOR).take(remaining));
                        return out;
                    }
                }
            }
        }
    }
    if current_a < current_b {
        out.push(B_INDICATOR);
    }
    // check if current value already is added
    if *out.last().unwrap() == A_INDICATOR {
        out.push(B_INDICATOR);
    }
    // take remaining
    out.extend(b_iter.map(|_| B_INDICATOR));
    assert_eq!(out.len(), b.len() + a.len());

    out
}

#[test]
fn test_foo() {
    let a = [1, 2, 4, 6, 9];
    let b = [2, 3, 4, 5, 10];

    let out = get_merge_indicator(&a, &b);
    let expected = [
        true, true, false, false, true, false, false, true, true, false,
    ];
    //                       1     2     2      3      4     4      5      6     9     10
    assert_eq!(out, expected);

    // swap
    // it is not the inverse because left is preferred when both are equal
    let out = get_merge_indicator(&b, &a);
    let b = [2, 3, 4, 5, 10];
    let a = [1, 2, 4, 6, 9];
    let expected = [
        false, true, false, true, true, false, true, false, false, true,
    ];
    assert_eq!(out, expected);

    let a = [5, 6, 7, 10];
    let b = [1, 2, 5];
    let out = get_merge_indicator(&a, &b);
    let expected = [false, false, true, false, true, true, true];
    assert_eq!(out, expected);

    // swap
    // it is not the inverse because left is preferred when both are equal
    let out = get_merge_indicator(&b, &a);
    let expected = [true, true, true, false, false, false, false];
    assert_eq!(out, expected);
}
