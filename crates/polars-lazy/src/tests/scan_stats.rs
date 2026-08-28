//! Assertions on the [`ScanStats`] resolved for a scan, per format.

use super::*;

/// A fresh directory under the system temp dir, removed by [`TempDir::drop`].
struct TempDir(std::path::PathBuf);

impl TempDir {
    fn new(name: &str) -> Self {
        let path = std::env::temp_dir().join(format!("polars-scan-stats-{name}"));
        _ = std::fs::remove_dir_all(&path);
        std::fs::create_dir_all(&path).unwrap();
        Self(path)
    }

    fn join(&self, name: &str) -> String {
        self.0.join(name).to_str().unwrap().to_owned()
    }
}

impl Drop for TempDir {
    fn drop(&mut self) {
        _ = std::fs::remove_dir_all(&self.0);
    }
}

fn scan_stats(q: LazyFrame) -> ScanStats {
    let (mut expr_arena, mut lp_arena) = get_arenas();
    let lp = q.optimize(&mut lp_arena, &mut expr_arena).unwrap();

    lp_arena
        .iter(lp)
        .find_map(|(_, lp)| match lp {
            IR::Scan { file_info, .. } => Some(file_info.stats.clone()),
            _ => None,
        })
        .expect("no scan in plan")
}

fn stats_df() -> DataFrame {
    df!(
        "a" => &[Some(1i32), None, Some(3), Some(3), None],
        "b" => &["xx", "yy", "zz", "xx", "yy"],
    )
    .unwrap()
}

#[cfg(feature = "parquet")]
fn write_parquet(dir: &TempDir, names: &[&str], statistics: StatisticsOptions) {
    for name in names {
        ParquetWriter::new(std::fs::File::create(dir.join(name)).unwrap())
            .with_statistics(statistics)
            .finish(&mut stats_df())
            .unwrap();
    }
}

#[cfg(feature = "ipc")]
fn write_ipc(dir: &TempDir, names: &[&str]) {
    for name in names {
        IpcWriter::new(std::fs::File::create(dir.join(name)).unwrap())
            .finish(&mut stats_df())
            .unwrap();
    }
}

#[cfg(feature = "parquet")]
fn scan_parquet_stats(path: &str, glob: bool) -> ScanStats {
    let args = ScanArgsParquet {
        glob,
        ..Default::default()
    };
    scan_stats(LazyFrame::scan_parquet(PlRefPath::new(path), args).unwrap())
}

#[cfg(feature = "ipc")]
fn scan_ipc_stats(path: &str, glob: bool) -> ScanStats {
    let args = UnifiedScanArgs {
        glob,
        ..Default::default()
    };
    scan_stats(LazyFrame::scan_ipc(PlRefPath::new(path), Default::default(), args).unwrap())
}

#[test]
#[cfg(feature = "parquet")]
fn test_parquet_scan_stats() {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let dir = TempDir::new("parquet-basic");
    write_parquet(&dir, &["a.parquet"], StatisticsOptions::full());

    let stats = scan_parquet_stats(&dir.join("a.parquet"), false);

    // A single fully resolved footer is a guarantee, not an estimate.
    assert_eq!(stats.rows, Card::Exact(5));

    let a = stats.column("a").expect("no stats for `a`");
    assert_eq!(a.null_count, Card::Exact(2));
    assert!(a.avg_byte_width.is_some_and(|w| w > 0.0));

    let b = stats.column("b").expect("no stats for `b`");
    assert_eq!(b.null_count, Card::Exact(0));
    assert!(b.avg_byte_width.is_some_and(|w| w > 0.0));

    // The writer does not emit `distinct_count`, and nothing may invent one.
    assert_eq!(a.distinct, Card::Unknown);
    assert_eq!(b.distinct, Card::Unknown);
}

#[test]
#[cfg(feature = "parquet")]
fn test_parquet_scan_stats_without_statistics() {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let dir = TempDir::new("parquet-nostats");
    write_parquet(&dir, &["a.parquet"], StatisticsOptions::empty());

    let stats = scan_parquet_stats(&dir.join("a.parquet"), false);

    // Row counts live in the footer regardless, but a null count without
    // statistics is unknown rather than zero.
    assert_eq!(stats.rows, Card::Exact(5));
    let a = stats.column("a").expect("no stats for `a`");
    assert_eq!(a.null_count, Card::Unknown);
    assert!(a.avg_byte_width.is_some_and(|w| w > 0.0));
}

#[test]
#[cfg(feature = "parquet")]
fn test_parquet_scan_stats_multi_source() {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let dir = TempDir::new("parquet-multi");
    write_parquet(&dir, &["a.parquet", "b.parquet"], StatisticsOptions::full());

    let stats = scan_parquet_stats(&dir.join("*.parquet"), true);

    assert_eq!(stats.rows, Card::Exact(10));
    let a = stats.column("a").expect("no stats for `a`");
    assert_eq!(a.null_count, Card::Exact(4));
}

#[test]
#[cfg(feature = "ipc")]
fn test_ipc_scan_stats_rows() {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let dir = TempDir::new("ipc-basic");
    write_ipc(&dir, &["a.ipc"]);

    // Previously thrown away: the footer blocks carry the record batch lengths.
    let stats = scan_ipc_stats(&dir.join("a.ipc"), false);
    assert_eq!(stats.rows, Card::Exact(5));
}

#[test]
#[cfg(feature = "ipc")]
fn test_ipc_scan_stats_multi_source_is_approx() {
    let _guard = SINGLE_LOCK.lock().unwrap();
    let dir = TempDir::new("ipc-multi");
    write_ipc(&dir, &["a.ipc", "b.ipc"]);

    // Only the first source is read, so the rest is extrapolated.
    let stats = scan_ipc_stats(&dir.join("*.ipc"), true);
    assert_eq!(stats.rows.value(), Some(10));
    assert!(matches!(stats.rows, Card::Approx { .. }));
}

#[test]
#[cfg(feature = "parquet")]
fn test_parquet_scan_stats_row_counts_mode() {
    const VAR: &str = "POLARS_RESOLVE_METADATA_LEVEL";
    let _guard = SINGLE_LOCK.lock().unwrap();

    let dir = TempDir::new("parquet-rowcounts");
    write_parquet(&dir, &["a.parquet", "b.parquet"], StatisticsOptions::full());

    unsafe { std::env::set_var(VAR, "row_counts") };
    polars_config::config().reload_env_var(VAR);

    let stats = scan_parquet_stats(&dir.join("*.parquet"), true);

    unsafe { std::env::remove_var(VAR) };
    polars_config::config().reload_env_var(VAR);

    // This mode reads every row count but retains only the first footer, so
    // the count is exact while the per-column fold covers one file of two.
    assert_eq!(stats.rows, Card::Exact(10));
    assert_eq!(
        stats.column("a").expect("no stats for `a`").null_count,
        Card::Unknown
    );
}
