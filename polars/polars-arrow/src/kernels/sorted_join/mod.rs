use crate::index::IdxSize;
use arrow::types::NativeType;
use std::fmt::Debug;

type LeftJoinIds = (JoinIds, JoinOptIds);
type JoinOptIds = Vec<Option<IdxSize>>;
type JoinIds = Vec<IdxSize>;


fn left<T>(lhs: &[T], rhs: &[T]) -> LeftJoinIds
where T: NativeType + PartialOrd + Debug
{
    if lhs.is_empty() {
        dbg!("return");
        return (vec![], vec![]);
    }
    if rhs.is_empty() {
        dbg!("return");
        return ((0..lhs.len() as IdxSize).collect(), vec![None; lhs.len()]);
    }
    let first_left = lhs[0];
    let first_right = rhs[0];

    // * 1.5 because there can be duplicates
    let cap = (lhs.len() as f32 * 1.5) as usize;
    let mut out_rhs = Vec::with_capacity(cap);
    let mut out_lhs = Vec::with_capacity(cap);

    dbg!(first_left, first_right);
    if first_left <= first_right {

        let mut last_right_offset = 0;
        for (lhs_i, lhs_val) in lhs.iter().enumerate() {

            // look for the value in the rhs
            let mut rhs_offset = last_right_offset;
            loop {
                dbg!(lhs_val, rhs.get(rhs_offset));
                match rhs.get(rhs_offset) {
                    Some(rhs_val) => {
                        if lhs_val < rhs_val {
                            out_lhs.push(lhs_i as IdxSize);
                            out_rhs.push(None);
                            dbg!("here");
                            // we break and must first increment left more
                            break;
                        }
                        // we found a match, we continue looping as there may be more
                        if lhs_val == rhs_val {
                            out_lhs.push(lhs_i as IdxSize);
                            out_rhs.push(Some(rhs_offset as IdxSize));
                            rhs_offset += 1;
                        }
                        // rhs is smaller than lhs
                        // we must increment
                        else {


                            // check if the next lhs value is the same as the current one
                            // if so we can continue from the same `last_right_offset`
                            // if not, we can increment the `last_right_offset` to `current_i`
                            match lhs.get(lhs_i + 1) {
                                Some(peek_lhs_val) => {
                                    if peek_lhs_val != lhs_val {
                                        last_right_offset = rhs_offset;
                                    }
                                    break;
                                }
                                // we depleted lhs, we can return
                                None => {
                                    dbg!("return");
                                    return (out_lhs, out_rhs)
                                }
                            }

                        }
                    }
                    // we depleted rhs, we can return
                    None => {
                        out_lhs.extend((lhs_i as IdxSize)..(lhs.len() as IdxSize));
                        out_rhs.extend(std::iter::repeat(None).take(lhs.len() - lhs_i));
                        return (out_lhs, out_rhs)
                    }
                }

            }
        }
    }
    out_rhs;
    todo!()
}


#[cfg(test)]
mod test {
    use super::*;

    #[test]
    fn test_left_join() {
        let lhs = &[0, 1, 1, 2, 3, 5];
        let rhs = &[0, 1, 1, 3, 4];

        let (l_idx, r_idx) = left(lhs, rhs);

        dbg!(l_idx, r_idx);

    }

}