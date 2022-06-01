use crate::index::IdxSize;
use std::fmt::Debug;

type LeftJoinIds = (JoinIds, JoinOptIds);
type JoinOptIds = Vec<Option<IdxSize>>;
type JoinIds = Vec<IdxSize>;

pub(super) fn left<T: PartialOrd + Copy + Debug>(left: &[T], right: &[T]) -> LeftJoinIds {
    if left.is_empty() {
        return (vec![], vec![]);
    }
    if right.is_empty() {
        return ((0..left.len() as IdxSize).collect(), vec![None; left.len()]);
    }
    // * 1.5 because there can be duplicates
    let cap = (left.len() as f32 * 1.5) as usize;
    let mut out_rhs = Vec::with_capacity(cap);
    let mut out_lhs = Vec::with_capacity(cap);

    let mut left_idx = 0 as IdxSize;
    let mut right_idx = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    // first values should be None, until left has catched up
    let mut left_catched_up = false;

    for &val_l in left {
        loop {
            match right.get(right_idx as usize) {
                Some(&val_r) => {
                    // we fill nulls until left value is larger than right
                    if !left_catched_up {
                        if val_l < val_r {
                            out_rhs.push(None);
                            out_lhs.push(left_idx);
                            break;
                        } else {
                            left_catched_up = true;
                        }
                    }

                    // matching join key
                    if val_l == val_r {
                        out_lhs.push(left_idx);
                        out_rhs.push(Some(right_idx));
                        let current_idx = right_idx;

                        loop {
                            right_idx += 1;
                            match right.get(right_idx as usize) {
                                // rhs depleted
                                None => {
                                    // reset right index because the next lhs value can be the same
                                    right_idx = current_idx;
                                    break;
                                }
                                Some(&val_r) => {
                                    if val_l == val_r {
                                        out_lhs.push(left_idx);
                                        out_rhs.push(Some(right_idx));
                                    } else {
                                        // reset right index because the next lhs value can be the same
                                        right_idx = current_idx;
                                        break;
                                    }
                                }
                            }
                        }
                        break;
                    }

                    // right is larger than left.
                    if val_r > val_l {
                        out_lhs.push(left_idx);
                        out_rhs.push(None);
                        break;
                    }
                    // continue looping the right side
                    right_idx += 1;
                }
                // we depleted the right array
                None => {
                    out_lhs.push(left_idx);
                    out_rhs.push(None);
                    break;
                }
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

        let (l_idx, r_idx) = left(lhs, rhs);

        assert_eq!(&l_idx, &[0, 1, 1, 2, 2, 3, 4, 5]);
        assert_eq!(
            &r_idx,
            &[
                Some(0),
                Some(1),
                Some(2),
                Some(1),
                Some(2),
                None,
                Some(3),
                None
            ]
        );

        let lhs = &[0, 0, 1, 3, 4, 5, 6, 6, 6, 7];
        let rhs = &[0, 0, 1, 3, 4, 6, 6];

        let (l_idx, r_idx) = left(lhs, rhs);
        assert_eq!(&l_idx, &[0, 0, 1, 1, 2, 3, 4, 5, 6, 6, 7, 7, 8, 8, 9]);
        assert_eq!(
            &r_idx,
            &[
                Some(0),
                Some(1),
                Some(0),
                Some(1),
                Some(2),
                Some(3),
                Some(4),
                None,
                Some(5),
                Some(6),
                Some(5),
                Some(6),
                Some(5),
                Some(6),
                None
            ]
        );

        let lhs = &[1, 3, 4, 5, 5, 5, 5, 6, 7, 7];
        let rhs = &[2, 4, 5, 6, 7, 8, 10, 11, 11, 12, 12, 12, 12, 13];
        let (l_idx, r_idx) = left(lhs, rhs);
        assert_eq!(&l_idx, &[0, 1, 2, 3, 4, 5, 6, 7, 8, 9]);
        assert_eq!(
            &r_idx,
            &[
                None,
                None,
                Some(1),
                Some(2),
                Some(2),
                Some(2),
                Some(2),
                Some(3),
                Some(4),
                Some(4)
            ]
        );
    }
}
