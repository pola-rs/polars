use super::*;

pub fn join<T: SortedJoinKey>(
    left: &[T],
    right: &[T],
    left_offset: IdxSize,
) -> InnerJoinIds {
    if left.is_empty() || right.is_empty() {
        return (vec![], vec![]);
    }

    // * 1.5 because of possible duplicates
    let cap = (std::cmp::min(left.len(), right.len()) as f32 * 1.5) as usize;
    let mut out_rhs = Vec::with_capacity(cap);
    let mut out_lhs = Vec::with_capacity(cap);

    let mut right_idx = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    let first_right = right[0];
    let mut left_idx = left.partition_point(|v| v.tot_lt(&first_right)) as IdxSize;

    #[allow(clippy::explicit_counter_loop)]
    for &val_l in &left[left_idx as usize..] {
        while let Some(&val_r) = right.get(right_idx as usize) {
            // matching join key
            if val_l.tot_eq(&val_r) {
                out_lhs.push(left_idx + left_offset);
                out_rhs.push(right_idx);
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
                        Some(&val_r) if val_l.tot_eq(&val_r) => {
                            out_lhs.push(left_idx + left_offset);
                            out_rhs.push(right_idx);
                        },
                        Some(_) => {
                            right_idx = current_idx;
                            break;
                        },
                    }
                }
                break;
            }

            // right is larger than left.
            if val_r.tot_gt(&val_l) {
                break;
            }
            // continue looping the right side
            right_idx += 1;
        }
        left_idx += 1;
    }
    (out_lhs, out_rhs)
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_inner_join_nan_keys() {
        // Polars sorts `NaN` after every number and alongside every other `NaN`, so a run of
        // them is a run of equal keys like any other. Driven by `PartialOrd` this matched
        // nothing at all, and a join on a sorted float key dropped every `NaN` row.
        let nan = f64::NAN;
        let lhs = &[1.0, nan, nan];
        let rhs = &[1.0, nan];

        let (l_idx, r_idx) = join(lhs, rhs, 0);
        assert_eq!(&l_idx, &[0, 1, 2]);
        assert_eq!(&r_idx, &[0, 1, 1]);

        // Only `NaN`, and on both sides: every pair matches.
        let (l_idx, r_idx) = join(&[nan, nan], &[nan, nan], 0);
        assert_eq!(&l_idx, &[0, 0, 1, 1]);
        assert_eq!(&r_idx, &[0, 1, 0, 1]);

        // A `NaN` on the right only: the numbers still find each other, and nothing matches it.
        let (l_idx, r_idx) = join(&[1.0, 2.0], &[2.0, nan], 0);
        assert_eq!(&l_idx, &[1]);
        assert_eq!(&r_idx, &[0]);
    }

    #[test]
    fn test_inner_join() {
        let lhs = &[0, 1, 1, 2, 3, 5];
        let rhs = &[0, 1, 1, 3, 4];

        let (l_idx, r_idx) = join(lhs, rhs, 0);

        assert_eq!(&l_idx, &[0, 1, 1, 2, 2, 4]);
        assert_eq!(&r_idx, &[0, 1, 2, 1, 2, 3]);

        let lhs = &[4, 4, 4, 4, 5, 6, 6, 7, 7, 7];
        let rhs = &[0, 1, 2, 3, 4, 4, 4, 6, 7, 7];
        let (l_idx, r_idx) = join(lhs, rhs, 0);

        assert_eq!(
            &l_idx,
            &[0, 0, 0, 1, 1, 1, 2, 2, 2, 3, 3, 3, 5, 6, 7, 7, 8, 8, 9, 9]
        );
        assert_eq!(
            &r_idx,
            &[4, 5, 6, 4, 5, 6, 4, 5, 6, 4, 5, 6, 7, 7, 8, 9, 8, 9, 8, 9]
        );
    }
}
