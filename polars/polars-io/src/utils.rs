#[cfg(any(feature = "ipc", feature = "parquet"))]
use crate::ArrowSchema;
use dirs::home_dir;
use polars_core::frame::DataFrame;
use std::path::{Path, PathBuf};

// used by python polars
pub fn resolve_homedir(path: &Path) -> PathBuf {
    // replace "~" with home directory
    if path.starts_with("~") {
        if let Some(homedir) = home_dir() {
            return homedir.join(path.strip_prefix("~").unwrap());
        }
    }

    path.into()
}

#[cfg(any(feature = "ipc", feature = "parquet"))]
pub(crate) fn apply_projection(schema: &ArrowSchema, projection: &[usize]) -> ArrowSchema {
    let fields = &schema.fields;
    let fields = projection
        .iter()
        .map(|idx| fields[*idx].clone())
        .collect::<Vec<_>>();
    ArrowSchema::from(fields)
}

/// Because of threading every row starts from `0` or from `offset`.
/// We must correct that so that they are monotonically increasing.
pub(crate) fn update_row_counts(dfs: &mut [(DataFrame, u32)]) {
    if !dfs.is_empty() {
        let mut previous = dfs[0].1;
        for (df, n_read) in &mut dfs[1..] {
            if let Some(s) = df.get_columns_mut().get_mut(0) {
                *s = &*s + previous;
            }
            previous = *n_read;
        }
    }
}

#[cfg(test)]
mod tests {
    use super::resolve_homedir;
    use std::path::PathBuf;

    #[cfg(not(target_os = "windows"))]
    #[test]
    fn test_resolve_homedir() {
        let paths: Vec<PathBuf> = vec![
            "~/dir1/dir2/test.csv".into(),
            "/abs/path/test.csv".into(),
            "rel/path/test.csv".into(),
            "/".into(),
            "~".into(),
        ];

        let resolved: Vec<PathBuf> = paths.iter().map(|x| resolve_homedir(x)).collect();

        assert_eq!(resolved[0].file_name(), paths[0].file_name());
        assert!(resolved[0].is_absolute());
        assert_eq!(resolved[1], paths[1]);
        assert_eq!(resolved[2], paths[2]);
        assert_eq!(resolved[3], paths[3]);
        assert!(resolved[4].is_absolute());
    }

    #[cfg(target_os = "windows")]
    #[test]
    fn test_resolve_homedir_windows() {
        let paths: Vec<PathBuf> = vec![
            r#"c:\Users\user1\test.csv"#.into(),
            r#"~\user1\test.csv"#.into(),
            "~".into(),
        ];

        let resolved: Vec<PathBuf> = paths.iter().map(|x| resolve_homedir(x)).collect();

        assert_eq!(resolved[0], paths[0]);
        assert_eq!(resolved[1].file_name(), paths[1].file_name());
        assert!(resolved[1].is_absolute());
        assert!(resolved[2].is_absolute());
    }
}
