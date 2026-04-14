use std::sync::Arc;

use polars_error::{PolarsError, PolarsResult, polars_err};
use polars_io::pl_async;
use polars_plan::dsl::sink::{SinkedFileInfo, SinkedFilesCallback, SinkedFilesCallbackArgs};
use polars_utils::pl_path::PlRefPath;

pub async fn call_sinked_files_callback(
    sinked_files_callback: SinkedFilesCallback,
    sinked_file_info_list: SinkedFileInfoList,
) -> PolarsResult<()> {
    let SinkedFileInfoList { file_info_list } = &sinked_file_info_list;

    file_info_list
        .lock()
        .sort_unstable_by(|l, r| PlRefPath::cmp(&l.path, &r.path));

    pl_async::get_runtime()
        .spawn_blocking(move || {
            let SinkedFileInfoList { file_info_list } = sinked_file_info_list;

            let args = SinkedFilesCallbackArgs {
                file_info_list: std::mem::take(&mut file_info_list.lock()),
            };

            sinked_files_callback.call_(args)
        })
        .await
        .unwrap()
}

#[derive(Default, Debug, Clone)]
pub struct SinkedFileInfoList {
    pub file_info_list: Arc<parking_lot::Mutex<Vec<SinkedFileInfo>>>,
}

impl SinkedFileInfoList {
    pub fn non_path_error() -> PolarsError {
        polars_err!(
            ComputeError:
            "files callback was set but encountered non-path sink target"
        )
    }
}
