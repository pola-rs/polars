use crate::trusted_len::TrustedLen;

pub trait FromIteratorReversed<T>: Sized {
    fn from_trusted_len_iter_rev<I: TrustedLen<Item = T>>(iter: I) -> Self;
}
