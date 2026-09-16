#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NumaRegionId(pub usize);

pub fn num_numa_regions() -> usize {
    1
}

pub fn cpu_idx_to_numa_region(_cpu_idx: usize) -> NumaRegionId {
    NumaRegionId(0)
}

pub fn pin_thread_to_numa_region(_region: NumaRegionId) {}
