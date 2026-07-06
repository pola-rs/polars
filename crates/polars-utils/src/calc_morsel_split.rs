use std::num::NonZeroU64;

#[derive(Default, Debug, Clone)]
pub struct PartSizesIter {
    base_part_size: u64,
    remaining_parts: usize,
    remainder_cutoff: usize,
}

impl PartSizesIter {
    pub fn new_from_total_size(total_size: u64, n_parts: usize) -> Self {
        if n_parts == 0 {
            return Default::default();
        }

        let base_part_size = total_size / n_parts as u64;
        let remainder = total_size % n_parts as u64;
        let remainder_cutoff = usize::try_from(n_parts as u64 - remainder).unwrap();

        Self {
            base_part_size,
            remaining_parts: n_parts,
            remainder_cutoff,
        }
    }

    pub fn new_from_part_size(part_size: u64, n_parts: usize) -> Self {
        Self {
            base_part_size: part_size,
            remaining_parts: n_parts,
            remainder_cutoff: usize::MAX,
        }
    }

    pub fn base_part_size(&self) -> u64 {
        self.base_part_size
    }
}

impl Iterator for PartSizesIter {
    type Item = u64;

    fn next(&mut self) -> Option<Self::Item> {
        self.remaining_parts = self.remaining_parts.checked_sub(1)?;
        Some(self.base_part_size + (self.remaining_parts >= self.remainder_cutoff) as u64)
    }

    fn size_hint(&self) -> (usize, Option<usize>) {
        let n = ExactSizeIterator::len(self);
        (n, Some(n))
    }
}

impl ExactSizeIterator for PartSizesIter {
    fn len(&self) -> usize {
        self.remaining_parts
    }
}

/// Number of parts to split `size` to minimize the average part size difference to `target_part_size`.
pub fn calc_n_parts(size: u64, target_part_size: NonZeroU64) -> u64 {
    if size <= target_part_size.get() {
        return if size == 0 { 0 } else { 1 };
    }

    let n_parts = size / target_part_size.get();

    (n_parts..=n_parts.saturating_add(1))
        .min_by_key(|n_parts| (size / *n_parts).abs_diff(target_part_size.get()))
        .unwrap()
}

#[cfg(test)]
mod tests {
    use std::num::NonZeroU64;

    use crate::calc_morsel_split::{PartSizesIter, calc_n_parts};
    use crate::itertools::Itertools;

    #[test]
    fn test_calc_n_parts() {
        let mut prev_n_parts: u64 = 0;

        let boundaries = (0..1000u64)
            .filter(|i| {
                let n_parts = calc_n_parts(*i, const { NonZeroU64::new(100).unwrap() });
                let changed = n_parts != prev_n_parts;
                prev_n_parts = n_parts;
                changed
            })
            .collect::<Vec<_>>();

        assert_eq!(boundaries, [1, 134, 242, 345, 448, 550, 651, 752, 855, 954]);
    }

    #[test]
    fn test_part_sizes_iter() {
        assert_eq!(
            PartSizesIter::new_from_total_size(0, 0).collect_vec(),
            &[] as &[u64]
        );
        assert_eq!(
            PartSizesIter::new_from_total_size(1, 0).collect_vec(),
            &[] as &[u64]
        );
        assert_eq!(PartSizesIter::new_from_total_size(0, 1).collect_vec(), &[0]);
        assert_eq!(PartSizesIter::new_from_total_size(1, 1).collect_vec(), &[1]);
        assert_eq!(
            PartSizesIter::new_from_total_size(1, 2).collect_vec(),
            &[1, 0]
        );
        assert_eq!(PartSizesIter::new_from_total_size(2, 1).collect_vec(), &[2]);

        assert_eq!(
            PartSizesIter::new_from_total_size(100, 2).collect_vec(),
            &[50, 50]
        );
        assert_eq!(
            PartSizesIter::new_from_total_size(101, 2).collect_vec(),
            &[51, 50]
        );
        assert_eq!(
            PartSizesIter::new_from_total_size(102, 2).collect_vec(),
            &[51, 51]
        );
        assert_eq!(
            PartSizesIter::new_from_total_size(103, 2).collect_vec(),
            &[52, 51]
        );
        assert_eq!(
            PartSizesIter::new_from_total_size(104, 2).collect_vec(),
            &[52, 52]
        );
    }
}
