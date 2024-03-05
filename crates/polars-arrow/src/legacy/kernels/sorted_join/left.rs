use super::*;

pub fn join<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
    left_offset: IdxSize,
) -> LeftJoinIds {
    if left.is_empty() {
        return (vec![], vec![]);
    }
    if right.is_empty() {
        return (
            (left_offset..left.len() as IdxSize + left_offset).collect(),
            vec![NullableIdxSize::null(); left.len()],
        );
    }
    // * 1.5 because there can be duplicates
    let cap = (left.len() as f32 * 1.5) as usize;
    let mut out_rhs = Vec::with_capacity(cap);
    let mut out_lhs = Vec::with_capacity(cap);

    let mut right_idx = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    // first values should be None, until left has caught up

    let first_right = right[right_idx as usize];
    let mut left_idx = left.partition_point(|v| v < &first_right) as IdxSize;
    out_rhs.extend(std::iter::repeat(NullableIdxSize::null()).take(left_idx as usize));
    out_lhs.extend(left_offset..(left_idx + left_offset));

    for &val_l in &left[left_idx as usize..] {
        loop {
            match right.get(right_idx as usize) {
                Some(&val_r) => {
                    // matching join key
                    if val_l == val_r {
                        out_lhs.push(left_idx + left_offset);
                        out_rhs.push(right_idx.into());
                        let current_idx = right_idx;

                        loop {
                            right_idx += 1;
                            match right.get(right_idx as usize) {
                                // rhs depleted
                                None => {
                                    // reset right index because the next lhs value can be the same
                                    right_idx = current_idx;
                                    break;
                                },
                                Some(&val_r) => {
                                    if val_l == val_r {
                                        out_lhs.push(left_idx + left_offset);
                                        out_rhs.push(right_idx.into());
                                    } else {
                                        // reset right index because the next lhs value can be the same
                                        right_idx = current_idx;
                                        break;
                                    }
                                },
                            }
                        }
                        break;
                    }

                    // right is larger than left.
                    if val_r > val_l {
                        out_lhs.push(left_idx + left_offset);
                        out_rhs.push(NullableIdxSize::null());
                        break;
                    }
                    // continue looping the right side
                    right_idx += 1;
                },
                // we depleted the right array
                None => {
                    out_lhs.push(left_idx + left_offset);
                    out_rhs.push(NullableIdxSize::null());
                    break;
                },
            }
        }
        left_idx += 1;
    }
    (out_lhs, out_rhs)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_left_join() {
        let lhs = &[0, 1, 1, 2, 3, 5];
        let rhs = &[0, 1, 1, 3, 4];

        let (l_idx, r_idx) = join(lhs, rhs, 0);
        let out_left = &[0, 1, 1, 2, 2, 3, 4, 5];
        let out_right = &[
            0.into(),
            1.into(),
            2.into(),
            1.into(),
            2.into(),
            NullableIdxSize::null(),
            3.into(),
            NullableIdxSize::null(),
        ];
        assert_eq!(&l_idx, out_left);
        assert_eq!(&r_idx, out_right);

        let offset = 2;
        let (l_idx, r_idx) = join(&lhs[offset..], rhs, offset as IdxSize);
        assert_eq!(l_idx, out_left[3..]);
        assert_eq!(r_idx, out_right[3..]);

        let offset = 3;
        let (l_idx, r_idx) = join(&lhs[offset..], rhs, offset as IdxSize);
        assert_eq!(l_idx, out_left[5..]);
        assert_eq!(r_idx, out_right[5..]);

        let lhs = &[0, 0, 1, 3, 4, 5, 6, 6, 6, 7];
        let rhs = &[0, 0, 1, 3, 4, 6, 6];

        let (l_idx, r_idx) = join(lhs, rhs, 0);
        assert_eq!(&l_idx, &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9]);
        assert_eq!(
            &r_idx,
            &[
                0.into(),
                1.into(),
                0.into(),
                1.into(),
                2.into(),
                3.into(),
                4.into(),
                NullableIdxSize::null(),
                5.into(),
                6.into(),
                5.into(),
                6.into(),
                5.into(),
                6.into(),
                NullableIdxSize::null(),
            ]
        );

        let lhs = &[1, 3, 4, 5, 5, 5, 5, 6, 7, 7];
        let rhs = &[2, 4, 5, 6, 7, 8, 10, 11, 11, 12, 12, 12, 12, 13];
        let (l_idx, r_idx) = join(lhs, rhs, 0);
        assert_eq!(&l_idx, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            &r_idx,
            &[
                NullableIdxSize::null(),
                NullableIdxSize::null(),
                1.into(),
                2.into(),
                2.into(),
                2.into(),
                2.into(),
                3.into(),
                4.into(),
                4.into()
            ]
        );
        let lhs = &[0, 1, 2, 2, 3, 4, 4, 6, 6, 7];
        let rhs = &[4, 4, 4, 8];
        let (l_idx, r_idx) = join(lhs, rhs, 0);
        assert_eq!(&l_idx, &[0, 1, 2, 3, 4, 5, 5, 5, 6, 6, 6, 7, 8, 9]);
        assert_eq!(
            &r_idx,
            &[
                NullableIdxSize::null(),
                NullableIdxSize::null(),
                NullableIdxSize::null(),
                NullableIdxSize::null(),
                NullableIdxSize::null(),
                0.into(),
                1.into(),
                2.into(),
                0.into(),
                1.into(),
                2.into(),
                NullableIdxSize::null(),
                NullableIdxSize::null(),
                NullableIdxSize::null(),
            ]
        )
    }
}
