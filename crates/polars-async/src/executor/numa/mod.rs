use std::cmp::Reverse;
use std::ops::Range;
use std::sync::LazyLock;

mod hwloc;

use hwloc::{CpuSet, Topology};

/*
    We use logical CPU indices, the first N map 1:1 to physical CPUs, the
    others aren't mapped to any particular one.

    We create one NumaRegion per physical CPU NUMA region, plus one logical
    NumaRegion that contains all CPU indices not in a physical region.
*/
struct NumaRegion {
    num_phys_cpus: usize,
    cpu_set: Option<CpuSet>,
    cpu_idxs: Range<usize>,
}

static TOPOLOGY: LazyLock<Option<Topology>> = LazyLock::new(|| {
    if !polars_config::config().numa_aware() {
        return None;
    }

    match Topology::new() {
        Ok(t) => Some(t),
        Err(e) => {
            if polars_config::config().verbose() {
                eprintln!("NUMA support disabled, could not load topology: {e}")
            }

            None
        },
    }
});

static NUMA_REGIONS: LazyLock<Vec<NumaRegion>> = LazyLock::new(|| {
    let cfg = polars_config::config();
    let mut regions: Vec<_> = if cfg.numa_mock_regions() > 0 {
        let topo = TOPOLOGY
            .as_ref()
            .expect("can not mock NUMA regions without topology available");
        let mut cpu_set = topo.cpuset();
        let mut cpus_left = cpu_set.weight();
        let max_threads_per_region = cpus_left
            .min(cfg.max_threads())
            .div_ceil(cfg.numa_mock_regions() as usize);

        let mut regions = Vec::new();
        while !cpu_set.is_empty() {
            let mut region_set = CpuSet::new();
            for _ in 0..cpus_left.min(max_threads_per_region) {
                let idx = cpu_set.first_set().unwrap();
                region_set.set(idx);
                cpu_set.unset(idx);
                cpus_left -= 1;
            }

            regions.push(NumaRegion {
                num_phys_cpus: region_set.weight(),
                cpu_set: Some(region_set),
                cpu_idxs: 0..0,
            });
        }

        regions
    } else if let Some(topo) = &*TOPOLOGY {
        topo.numa_node_cpusets()
            .into_iter()
            .map(|cpu_set| NumaRegion {
                num_phys_cpus: cpu_set.weight(),
                cpu_set: Some(cpu_set),
                cpu_idxs: 0..0,
            })
            .collect()
    } else {
        Vec::new()
    };

    // Biggest NUMA region(s) as lower indices so we use those first.
    regions.sort_by_key(|r| Reverse(r.num_phys_cpus));

    // Assign ranges and add dummy infinite region afterwards.
    let mut start = 0;
    for r in &mut regions {
        r.cpu_idxs = start..start + r.num_phys_cpus;
        start = r.cpu_idxs.end;
    }
    regions.push(NumaRegion {
        num_phys_cpus: 0,
        cpu_set: None,
        cpu_idxs: start..usize::MAX,
    });
    regions
});

#[derive(Copy, Clone, PartialEq, Eq, Hash)]
pub struct NumaRegionId(pub usize);

pub fn num_numa_regions() -> usize {
    NUMA_REGIONS.len()
}

pub fn cpu_idx_to_numa_region(cpu_idx: usize) -> NumaRegionId {
    for (r_idx, r) in NUMA_REGIONS.iter().enumerate() {
        if r.cpu_idxs.contains(&cpu_idx) {
            return NumaRegionId(r_idx);
        }
    }

    unreachable!()
}

pub fn pin_thread_to_numa_region(region: NumaRegionId) {
    let Some(topo) = &*TOPOLOGY else {
        return;
    };

    let Some(cpu_set) = &NUMA_REGIONS[region.0].cpu_set else {
        return;
    };

    if let Err(e) = topo.bind_thread(cpu_set) {
        if polars_config::config().verbose() {
            eprintln!("can't bind thread to NUMA region: {e}")
        }
    }
}
