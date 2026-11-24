use polars_arrow::array::builder::{ShareStrategy, StaticArrayBuilder};
use polars_arrow::array::{Array, ChildPositions, LargeUnionArray, LargeUnionArrayBuilder};
use polars_arrow::bitmap::{Bitmap, BitmapBuilder};

use super::TotalEqKernel;
use super::dyn_array::array_tot_eq_missing_kernel;
use crate::gather::gather_child;

/// Calls `f(slot, variant, lhs_pos, rhs_pos)` for each slot valid on both sides with the same
/// variant.
fn for_each_comparable_slot(
    lhs: &LargeUnionArray,
    rhs: &LargeUnionArray,
    mut f: impl FnMut(usize, usize, usize, usize),
) {
    let (lv, rv) = (lhs.validity(), rhs.validity());
    for i in 0..lhs.len() {
        // SAFETY: `i` is in bounds of both arrays.
        unsafe {
            if lv.is_some_and(|v| !v.get_bit_unchecked(i))
                || rv.is_some_and(|v| !v.get_bit_unchecked(i))
            {
                continue;
            }
            let (lt, lo) = lhs.index_unchecked(i);
            let (rt, ro) = rhs.index_unchecked(i);
            if lt == rt {
                f(i, lt, lo, ro);
            }
        }
    }
}

impl TotalEqKernel for LargeUnionArray {
    type Scalar = Box<dyn Array>;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        assert_eq!(self.len(), other.len());
        assert_eq!(self.dtype(), other.dtype());

        // Align the compared values of each variant and compare them in one pass per child.
        let new_positions = |a: &LargeUnionArray| -> Vec<ChildPositions> {
            a.fields()
                .iter()
                .map(|c| ChildPositions::new(c.len()))
                .collect()
        };
        let mut lhs_pos = new_positions(self);
        let mut rhs_pos = new_positions(other);
        for_each_comparable_slot(self, other, |_, t, lo, ro| {
            lhs_pos[t].push(Some(lo));
            rhs_pos[t].push(Some(ro));
        });
        let child_eq: Vec<Bitmap> = self
            .fields()
            .iter()
            .zip(other.fields())
            .zip(lhs_pos.into_iter().zip(rhs_pos))
            .map(|((l, r), (lp, rp))| {
                // SAFETY: the positions of valid slots are in bounds.
                let (l, r) =
                    unsafe { (gather_child(l.as_ref(), lp), gather_child(r.as_ref(), rp)) };
                array_tot_eq_missing_kernel(l.as_ref(), r.as_ref())
            })
            .collect();

        let mut out = BitmapBuilder::with_capacity(self.len());
        let mut next = vec![0usize; child_eq.len()];
        for_each_comparable_slot(self, other, |i, t, _, _| {
            out.extend_constant(i - out.len(), false);
            out.push(child_eq[t].get_bit(next[t]));
            next[t] += 1;
        });
        out.extend_constant(self.len() - out.len(), false);
        out.freeze()
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        !&self.tot_eq_kernel(other)
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.tot_eq_kernel(&repeat_scalar(other.as_ref(), self.len()))
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.tot_ne_kernel(&repeat_scalar(other.as_ref(), self.len()))
    }
}

fn repeat_scalar(scalar: &dyn Array, len: usize) -> LargeUnionArray {
    let scalar: &LargeUnionArray = scalar.as_any().downcast_ref().unwrap();
    assert_eq!(scalar.len(), 1);
    let mut builder = LargeUnionArrayBuilder::new(scalar.dtype().clone());
    builder.subslice_extend_each_repeated(scalar, 0, 1, len, ShareStrategy::Always);
    builder.freeze()
}
