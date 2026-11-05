use std::ffi::c_uint;
use std::ptr::NonNull;

use hwlocality_sys::{
    HWLOC_CPUBIND_THREAD, HWLOC_OBJ_NUMANODE, hwloc_bitmap_alloc, hwloc_bitmap_clr,
    hwloc_bitmap_dup, hwloc_bitmap_first, hwloc_bitmap_free, hwloc_bitmap_iszero, hwloc_bitmap_s,
    hwloc_bitmap_set, hwloc_bitmap_weight, hwloc_const_cpuset_t, hwloc_get_nbobjs_by_depth,
    hwloc_get_obj_by_depth, hwloc_get_type_depth, hwloc_set_cpubind, hwloc_topology,
    hwloc_topology_destroy, hwloc_topology_get_topology_cpuset, hwloc_topology_init,
    hwloc_topology_load, hwloc_topology_t,
};

#[derive(Debug)]
pub struct HwlocError {
    api: &'static str,
    io_err: std::io::Error,
}

impl std::fmt::Display for HwlocError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        write!(
            f,
            "hwloc call '{}' failed with code: {}",
            self.api, self.io_err
        )
    }
}

impl std::error::Error for HwlocError {}

impl HwlocError {
    fn last(api: &'static str) -> Self {
        Self {
            api,
            io_err: std::io::Error::last_os_error(),
        }
    }
}

/// Owned hwloc topology handle.
pub struct Topology(NonNull<hwloc_topology>);

unsafe impl Send for Topology {}
unsafe impl Sync for Topology {}

impl Topology {
    // Create a new topology.
    pub fn new() -> Result<Self, HwlocError> {
        unsafe {
            let mut topo_ptr: hwloc_topology_t = std::ptr::null_mut();
            if hwloc_topology_init(&raw mut topo_ptr) != 0 {
                return Err(HwlocError::last("hwloc_topology_init"));
            }

            let slf = Self(NonNull::new(topo_ptr).unwrap());
            if hwloc_topology_load(slf.0.as_ptr()) != 0 {
                return Err(HwlocError::last("hwloc_topology_load"));
            }
            Ok(slf)
        }
    }

    /// The set of CPUs covered by this topology.
    pub fn cpuset(&self) -> CpuSet {
        // SAFETY: hwloc_topology_get_topology_cpuset can't return null.
        unsafe { CpuSet::dup_from_raw(hwloc_topology_get_topology_cpuset(self.0.as_ptr())) }
    }

    /// The owned cpuset of every NUMA node object in the topology.
    pub fn numa_node_cpusets(&self) -> Vec<CpuSet> {
        unsafe {
            let depth = hwloc_get_type_depth(self.0.as_ptr(), HWLOC_OBJ_NUMANODE);
            (0..hwloc_get_nbobjs_by_depth(self.0.as_ptr(), depth))
                .filter_map(|idx| {
                    let obj_ptr = hwloc_get_obj_by_depth(self.0.as_ptr(), depth, idx);
                    if obj_ptr.is_null() || (*obj_ptr).cpuset.is_null() {
                        return None;
                    }
                    Some(CpuSet::dup_from_raw((*obj_ptr).cpuset))
                })
                .collect()
        }
    }

    /// Bind the current thread to the given cpuset.
    pub fn bind_thread(&self, cpu_set: &CpuSet) -> Result<(), HwlocError> {
        unsafe {
            if hwloc_set_cpubind(self.0.as_ptr(), cpu_set.0.as_ptr(), HWLOC_CPUBIND_THREAD) != 0 {
                return Err(HwlocError::last("hwloc_set_cpubind"));
            }
        }

        Ok(())
    }
}

impl Drop for Topology {
    fn drop(&mut self) {
        unsafe { hwloc_topology_destroy(self.0.as_ptr()) }
    }
}

/// Owned hwloc CPU bitmap.
pub struct CpuSet(NonNull<hwloc_bitmap_s>);

unsafe impl Send for CpuSet {}
unsafe impl Sync for CpuSet {}

impl CpuSet {
    /// Creates a new empty bitmap.
    pub fn new() -> Self {
        let ptr = unsafe { hwloc_bitmap_alloc() };
        Self(NonNull::new(ptr).unwrap())
    }

    /// Clone the bitmap behind an hwloc cpuset pointer into an owned `CpuSet`.
    ///
    /// # Safety
    /// `ptr` must be non-null and point to a valid hwloc bitmap.
    unsafe fn dup_from_raw(ptr: hwloc_const_cpuset_t) -> Self {
        let dup = unsafe { hwloc_bitmap_dup(ptr) };
        Self(NonNull::new(dup).unwrap())
    }

    /// Number of CPUs set. A topology cpuset is always finite.
    pub fn weight(&self) -> usize {
        // SAFETY: valid bitmap.
        let weight = unsafe { hwloc_bitmap_weight(self.0.as_ptr()) };
        usize::try_from(weight).unwrap()
    }

    pub fn is_empty(&self) -> bool {
        unsafe { hwloc_bitmap_iszero(self.0.as_ptr()) != 0 }
    }

    /// Index of the first set CPU, or `None` if empty.
    pub fn first_set(&self) -> Option<usize> {
        let first = unsafe { hwloc_bitmap_first(self.0.as_ptr()) };
        usize::try_from(first).ok()
    }

    pub fn set(&mut self, idx: usize) {
        let _ = unsafe { hwloc_bitmap_set(self.0.as_ptr(), idx as c_uint) };
    }

    pub fn unset(&mut self, idx: usize) {
        let _ = unsafe { hwloc_bitmap_clr(self.0.as_ptr(), idx as c_uint) };
    }
}

impl Drop for CpuSet {
    fn drop(&mut self) {
        unsafe { hwloc_bitmap_free(self.0.as_ptr()) }
    }
}
