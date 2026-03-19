use arrow::array::{BinaryViewArray, Utf8ViewArray};
use arrow::bitmap::Bitmap;

use super::TotalEqKernel;
use crate::comparisons::TotalOrdKernel;

// If s fits in 12 bytes, returns the view encoding it would have in a
// BinaryViewArray.
fn small_view_encoding(s: &[u8]) -> Option<u128> {
    if s.len() > 12 {
        return None;
    }

    let mut tmp = [0u8; 16];
    tmp[0] = s.len() as u8;
    tmp[4..4 + s.len()].copy_from_slice(s);
    Some(u128::from_le_bytes(tmp))
}

// Loads (up to) the first 4 bytes of s as little-endian, padded with zeros.
fn load_prefix(s: &[u8]) -> u32 {
    let start = &s[..s.len().min(4)];
    let mut tmp = [0u8; 4];
    tmp[..start.len()].copy_from_slice(start);
    u32::from_le_bytes(tmp)
}

fn broadcast_inequality(
    arr: &BinaryViewArray,
    scalar: &[u8],
    cmp_prefix: impl Fn(u32, u32) -> bool,
    cmp_str: impl Fn(&[u8], &[u8]) -> bool,
) -> Bitmap {
    let views = arr.views().as_slice();
    let prefix = load_prefix(scalar);
    let be_prefix = prefix.to_be();
    Bitmap::from_trusted_len_iter((0..arr.len()).map(|i| unsafe {
        let v_prefix = (views.get_unchecked(i).as_u128() >> 32) as u32;
        if v_prefix != prefix {
            cmp_prefix(v_prefix.to_be(), be_prefix)
        } else {
            cmp_str(arr.value_unchecked(i), scalar)
        }
    }))
}

impl TotalEqKernel for BinaryViewArray {
    type Scalar = [u8];

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        debug_assert!(self.len() == other.len());

        let slf_views = self.views().as_slice();
        let other_views = other.views().as_slice();

        Bitmap::from_trusted_len_iter((0..self.len()).map(|i| unsafe {
            let av = slf_views.get_unchecked(i).as_u128();
            let bv = other_views.get_unchecked(i).as_u128();

            // First 64 bits contain length and prefix.
            let a_len_prefix = av as u64;
            let b_len_prefix = bv as u64;
            if a_len_prefix != b_len_prefix {
                return false;
            }

            let alen = av as u32;
            if alen <= 12 {
                // String is fully inlined, compare top 64 bits. Bottom bits were
                // tested equal before, which also ensures the lengths are equal.
                (av >> 64) as u64 == (bv >> 64) as u64
            } else {
                self.value_unchecked(i) == other.value_unchecked(i)
            }
        }))
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        debug_assert!(self.len() == other.len());

        let slf_views = self.views().as_slice();
        let other_views = other.views().as_slice();

        Bitmap::from_trusted_len_iter((0..self.len()).map(|i| unsafe {
            let av = slf_views.get_unchecked(i).as_u128();
            let bv = other_views.get_unchecked(i).as_u128();

            // First 64 bits contain length and prefix.
            let a_len_prefix = av as u64;
            let b_len_prefix = bv as u64;
            if a_len_prefix != b_len_prefix {
                return true;
            }

            let alen = av as u32;
            if alen <= 12 {
                // String is fully inlined, compare top 64 bits. Bottom bits were
                // tested equal before, which also ensures the lengths are equal.
                (av >> 64) as u64 != (bv >> 64) as u64
            } else {
                self.value_unchecked(i) != other.value_unchecked(i)
            }
        }))
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if let Some(val) = small_view_encoding(other) {
            Bitmap::from_trusted_len_iter(self.views().iter().map(|v| v.as_u128() == val))
        } else {
            let slf_views = self.views().as_slice();
            let prefix = u32::from_le_bytes(other[..4].try_into().unwrap());
            let prefix_len = ((prefix as u64) << 32) | other.len() as u64;
            Bitmap::from_trusted_len_iter((0..self.len()).map(|i| unsafe {
                let v_prefix_len = slf_views.get_unchecked(i).as_u128() as u64;
                if v_prefix_len != prefix_len {
                    false
                } else {
                    self.value_unchecked(i) == other
                }
            }))
        }
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        if let Some(val) = small_view_encoding(other) {
            Bitmap::from_trusted_len_iter(self.views().iter().map(|v| v.as_u128() != val))
        } else {
            let slf_views = self.views().as_slice();
            let prefix = u32::from_le_bytes(other[..4].try_into().unwrap());
            let prefix_len = ((prefix as u64) << 32) | other.len() as u64;
            Bitmap::from_trusted_len_iter((0..self.len()).map(|i| unsafe {
                let v_prefix_len = slf_views.get_unchecked(i).as_u128() as u64;
                if v_prefix_len != prefix_len {
                    true
                } else {
                    self.value_unchecked(i) != other
                }
            }))
        }
    }
}

impl TotalOrdKernel for BinaryViewArray {
    type Scalar = [u8];

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        debug_assert!(self.len() == other.len());

        let slf_views = self.views().as_slice();
        let other_views = other.views().as_slice();

        Bitmap::from_trusted_len_iter((0..self.len()).map(|i| unsafe {
            let av = slf_views.get_unchecked(i).as_u128();
            let bv = other_views.get_unchecked(i).as_u128();

            // First 64 bits contain length and prefix.
            // Only check prefix.
            let a_prefix = (av >> 32) as u32;
            let b_prefix = (bv >> 32) as u32;
            if a_prefix != b_prefix {
                a_prefix.to_be() < b_prefix.to_be()
            } else {
                self.value_unchecked(i) < other.value_unchecked(i)
            }
        }))
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        debug_assert!(self.len() == other.len());

        let slf_views = self.views().as_slice();
        let other_views = other.views().as_slice();

        Bitmap::from_trusted_len_iter((0..self.len()).map(|i| unsafe {
            let av = slf_views.get_unchecked(i).as_u128();
            let bv = other_views.get_unchecked(i).as_u128();

            // First 64 bits contain length and prefix.
            // Only check prefix.
            let a_prefix = (av >> 32) as u32;
            let b_prefix = (bv >> 32) as u32;
            if a_prefix != b_prefix {
                a_prefix.to_be() < b_prefix.to_be()
            } else {
                self.value_unchecked(i) <= other.value_unchecked(i)
            }
        }))
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        broadcast_inequality(self, other, |a, b| a < b, |a, b| a < b)
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        broadcast_inequality(self, other, |a, b| a <= b, |a, b| a <= b)
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        broadcast_inequality(self, other, |a, b| a > b, |a, b| a > b)
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        broadcast_inequality(self, other, |a, b| a >= b, |a, b| a >= b)
    }
}

impl TotalEqKernel for Utf8ViewArray {
    type Scalar = str;

    fn tot_eq_kernel(&self, other: &Self) -> Bitmap {
        self.to_binview().tot_eq_kernel(&other.to_binview())
    }

    fn tot_ne_kernel(&self, other: &Self) -> Bitmap {
        self.to_binview().tot_ne_kernel(&other.to_binview())
    }

    fn tot_eq_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binview().tot_eq_kernel_broadcast(other.as_bytes())
    }

    fn tot_ne_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binview().tot_ne_kernel_broadcast(other.as_bytes())
    }
}

impl TotalOrdKernel for Utf8ViewArray {
    type Scalar = str;

    fn tot_lt_kernel(&self, other: &Self) -> Bitmap {
        self.to_binview().tot_lt_kernel(&other.to_binview())
    }

    fn tot_le_kernel(&self, other: &Self) -> Bitmap {
        self.to_binview().tot_le_kernel(&other.to_binview())
    }

    fn tot_lt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binview().tot_lt_kernel_broadcast(other.as_bytes())
    }

    fn tot_le_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binview().tot_le_kernel_broadcast(other.as_bytes())
    }

    fn tot_gt_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binview().tot_gt_kernel_broadcast(other.as_bytes())
    }

    fn tot_ge_kernel_broadcast(&self, other: &Self::Scalar) -> Bitmap {
        self.to_binview().tot_ge_kernel_broadcast(other.as_bytes())
    }
}
