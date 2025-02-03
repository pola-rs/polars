use polars_core::prelude::*;
use polars_core::POOL;

fn mode_indices(groups: GroupsType) -> Vec<IdxSize> {
    match groups {
        GroupsType::Idx(groups) => {
            let Some(max_len) = groups.iter().map(|g| g.1.len()).max() else {
                return Vec::new();
            };
            groups
                .into_iter()
                .filter(|g| g.1.len() == max_len)
                .map(|g| g.0)
                .collect()
        },
        GroupsType::Slice { groups, .. } => {
            let Some(max_len) = groups.iter().map(|g| g[1]).max() else {
                return Vec::new();
            };
            groups
                .into_iter()
                .filter(|g| g[1] == max_len)
                .map(|g| g[0])
                .collect()
        },
    }
}

pub fn mode(s: &Series) -> PolarsResult<Series> {
    let parallel = !POOL.current_thread_has_pending_tasks().unwrap_or(false);
    let groups = s.group_tuples(parallel, false).unwrap();
    let idx = mode_indices(groups);
    let idx = IdxCa::from_vec("".into(), idx);
    // SAFETY:
    // group indices are in bounds
    Ok(unsafe { s.take_unchecked(&idx) })
}
