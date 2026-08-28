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

#[test]
#[cfg(feature = "parquet")]
fn test_parquet_scan_stats() {
    let dir = TempDir::new("parquet-basic");
    let path = dir.join("a.parquet");
    let mut df = stats_df();
    ParquetWriter::new(std::fs::File::create(&path).unwrap())
        .with_statistics(StatisticsOptions::full())
        .finish(&mut df)
        .unwrap();

    let stats = scan_stats(
        LazyFrame::scan_parquet(PlRefPath::new(path.as_str()), Default::default()).unwrap(),
    );

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
    let dir = TempDir::new("parquet-nostats");
    let path = dir.join("a.parquet");
    let mut df = stats_df();
    ParquetWriter::new(std::fs::File::create(&path).unwrap())
        .with_statistics(StatisticsOptions::empty())
        .finish(&mut df)
        .unwrap();

    let stats = scan_stats(
        LazyFrame::scan_parquet(PlRefPath::new(path.as_str()), Default::default()).unwrap(),
    );

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
    let dir = TempDir::new("parquet-multi");
    for name in ["a.parquet", "b.parquet"] {
        let mut df = stats_df();
        ParquetWriter::new(std::fs::File::create(dir.join(name)).unwrap())
            .with_statistics(StatisticsOptions::full())
            .finish(&mut df)
            .unwrap();
    }

    let glob = dir.join("*.parquet");
    let stats = scan_stats(
        LazyFrame::scan_parquet(
            PlRefPath::new(glob.as_str()),
            ScanArgsParquet {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap(),
    );

    assert_eq!(stats.rows, Card::Exact(10));
    let a = stats.column("a").expect("no stats for `a`");
    assert_eq!(a.null_count, Card::Exact(4));
}

#[test]
#[cfg(feature = "ipc")]
fn test_ipc_scan_stats_rows() {
    let dir = TempDir::new("ipc-basic");
    let path = dir.join("a.ipc");
    let mut df = stats_df();
    IpcWriter::new(std::fs::File::create(&path).unwrap())
        .finish(&mut df)
        .unwrap();

    // Previously thrown away: the footer blocks carry the record batch lengths.
    let stats = scan_stats(
        LazyFrame::scan_ipc(
            PlRefPath::new(path.as_str()),
            Default::default(),
            Default::default(),
        )
        .unwrap(),
    );
    assert_eq!(stats.rows, Card::Exact(5));
}

#[test]
#[cfg(feature = "ipc")]
fn test_ipc_scan_stats_multi_source_is_approx() {
    let dir = TempDir::new("ipc-multi");
    for name in ["a.ipc", "b.ipc"] {
        let mut df = stats_df();
        IpcWriter::new(std::fs::File::create(dir.join(name)).unwrap())
            .finish(&mut df)
            .unwrap();
    }

    // Only the first source is read, so the rest is extrapolated.
    let glob = dir.join("*.ipc");
    let stats = scan_stats(
        LazyFrame::scan_ipc(
            PlRefPath::new(glob.as_str()),
            Default::default(),
            UnifiedScanArgs {
                glob: true,
                ..Default::default()
            },
        )
        .unwrap(),
    );
    assert_eq!(stats.rows.value(), Some(10));
    assert!(matches!(stats.rows, Card::Approx { .. }));
}
