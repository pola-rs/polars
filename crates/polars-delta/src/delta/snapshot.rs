//! Owned boundary between snapshot planning and the Polars file-table exporter.
//! Paths in this interface are raw Delta add paths. File indices are in global
//! raw-path order. Sources preserve absent DV offsets, partition nulls and
//! optional statistics until export. No callback or storage handle may escape here.
use serde_json::Value;

use super::files::Source;

/// One version of the table. File actions are left where they were read: `files` lists the
/// surviving adds as `(source, row)`, sorted by path. A snapshot read without files has none.
pub struct Snapshot {
    pub version: i64,
    pub metadata: Value,
    pub protocol: Value,
    pub sources: Vec<Source>,
    pub files: Vec<(u32, u32)>,
}
impl Snapshot {
    pub fn file_at_path(&self, path: &str) -> Option<(&Source, usize)> {
        let index = self
            .files
            .binary_search_by(|&(source, row)| {
                self.sources[source as usize].path(row as usize).cmp(path)
            })
            .ok()?;
        let (source, row) = self.files[index];
        Some((&self.sources[source as usize], row as usize))
    }
}
