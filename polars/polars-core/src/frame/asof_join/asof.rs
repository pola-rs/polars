use polars_arrow::index::IdxSize;
use std::fmt::Debug;

pub(super) fn join_asof_backward<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());

    let mut offset = 0 as IdxSize;
    // left array could start lower than right;
    // left: [-1, 0, 1, 2],
    // right: [1, 2, 3]
    // first values should be None, until left has catched up
    let mut left_catched_up = false;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    if !left_catched_up {
                        if val_l < val_r {
                            out.push(None);
                            break;
                        } else {
                            left_catched_up = true;
                        }
                    }

                    // the branch where
                    if val_r > val_l {
                        out.push(Some(offset - 1));
                        break;
                    } else {
                        offset += 1;
                    }
                }
                None => {
                    let val = if left_catched_up {
                        Some(offset - 1)
                    } else {
                        None
                    };
                    out.extend(std::iter::repeat(val).take(left.len() - out.len()));
                    return out;
                }
            }
        }
    }
    out
}

pub(super) fn join_asof_forward<T: PartialOrd + Copy + Debug>(
    left: &[T],
    right: &[T],
) -> Vec<Option<IdxSize>> {
    let mut out = Vec::with_capacity(left.len());
    let mut offset = 0 as IdxSize;

    for &val_l in left {
        loop {
            match right.get(offset as usize) {
                Some(&val_r) => {
                    if val_r >= val_l {
                        out.push(Some(offset));
                        break;
                    }
                    offset += 1;
                }
                None => {
                    out.extend(std::iter::repeat(None).take(left.len() - out.len()));
                    return out;
                }
            }
        }
    }
    out
}

#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_asof_backward() {
        let a = [-1, 2, 3, 3, 3, 4];
        let b = [1, 2, 3, 3];

        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples, &[None, Some(1), Some(3), Some(3), Some(3), Some(3)]);

        let b = [1, 2, 4, 5];
        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples, &[None, Some(1), Some(1), Some(1), Some(1), Some(2)]);

        let a = [2, 4, 4, 4];
        let b = [1, 2, 3, 3];
        let tuples = join_asof_backward(&a, &b);
        assert_eq!(tuples, &[Some(1), Some(3), Some(3), Some(3)]);
    }

    #[test]
    fn test_asof_forward() {
        let a = [-1, 1, 2, 4, 6];
        let b = [1, 2, 4, 5];

        let tuples = join_asof_forward(&a, &b);
        assert_eq!(tuples.len(), a.len());
        assert_eq!(tuples, &[Some(0), Some(0), Some(1), Some(2), None]);
    }
}
