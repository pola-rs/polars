use std::sync::Arc;

use polars_core::runtime::ASYNC;
use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_plan::dsl::sink::{SinkedPathInfo, SinkedPathsCallback, SinkedPathsCallbackArgs};
use polars_utils::pl_path::PlRefPath;

pub async fn call_sinked_paths_callback(
    sinked_paths_callback: SinkedPathsCallback,
    sinked_path_info_list: SinkedPathInfoList,
) -> PolarsResult<()> {
    let SinkedPathInfoList { path_info_list } = &sinked_path_info_list;

    path_info_list.lock().sort_unstable_by(
        |SinkedPathInfo { path: l, .. }, SinkedPathInfo { path: r, .. }| PlRefPath::cmp(l, r),
    );

    ASYNC
        .spawn_blocking(move || {
            let SinkedPathInfoList { path_info_list } = sinked_path_info_list;

            let args = SinkedPathsCallbackArgs {
                path_info_list: std::mem::take(&mut path_info_list.lock()),
            };

            sinked_paths_callback.call(args)
        })
        .await
        .unwrap()
}

#[derive(Default, Debug, Clone)]
pub struct SinkedPathInfoList {
    path_info_list: Arc<parking_lot::Mutex<Vec<SinkedPathInfo>>>,
}

impl SinkedPathInfoList {
    pub fn new_entry(&self) -> SinkedPathInfoEntry {
        let mut v = self.path_info_list.lock();
        let entry_idx = v.len();
        v.push(SinkedPathInfo::default());

        SinkedPathInfoEntry {
            path_info_list: self.clone(),
            entry_idx,
        }
    }
}

pub fn requested_sinked_paths_callback_with_non_path_error() -> PolarsError {
    polars_err!(
        ComputeError:
        "paths callback was set but encountered non-path sink target"
    )
}

#[derive(Clone)]
pub struct SinkedPathInfoEntry {
    path_info_list: SinkedPathInfoList,
    entry_idx: usize,
}

impl SinkedPathInfoEntry {
    pub fn set_path(&self, path: PlRefPath) {
        self.path_info_list.path_info_list.lock()[self.entry_idx].path = path;
    }

    pub fn set_num_rows(&self, num_rows: u64) {
        self.path_info_list.path_info_list.lock()[self.entry_idx].num_rows = num_rows;
    }

    pub fn set_num_bytes(&self, num_bytes: u64) {
        self.path_info_list.path_info_list.lock()[self.entry_idx].num_bytes = num_bytes;
    }
}
