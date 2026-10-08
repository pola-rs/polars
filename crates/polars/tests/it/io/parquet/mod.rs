#![forbid(unsafe_code)]
mod arrow;
pub(crate) mod read;
mod roundtrip;
mod write;

use std::io::Cursor;
use std::path::PathBuf;

use polars::prelude::*;

// The dynamic representation of values in native Rust. This is not exhaustive.
// todo: maybe refactor this into serde/json?
#[derive(Debug, PartialEq)]
pub enum Array {
    Int32(Vec<Option<i32>>),
    Int64(Vec<Option<i64>>),
    Int96(Vec<Option<[u32; 3]>>),
    Float(Vec<Option<f32>>),
    Double(Vec<Option<f64>>),
    Boolean(Vec<Option<bool>>),
    Binary(Vec<Option<Vec<u8>>>),
    FixedLenBinary(Vec<Option<Vec<u8>>>),
    List(Vec<Option<Array>>),
    Struct(Vec<Array>, Vec<bool>),
}

use polars_parquet::parquet::schema::types::{PhysicalType, PrimitiveType};
use polars_parquet::parquet::statistics::*;

pub fn alltypes_plain(column: &str) -> Array {
    match column {
        "id" => {
            let expected = vec![4, 5, 6, 7, 2, 3, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "id-short-array" => {
            let expected = vec![4];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "bool_col" => {
            let expected = vec![true, false, true, false, true, false, true, false];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Boolean(expected)
        },
        "tinyint_col" => {
            let expected = vec![0, 1, 0, 1, 0, 1, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "smallint_col" => {
            let expected = vec![0, 1, 0, 1, 0, 1, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "int_col" => {
            let expected = vec![0, 1, 0, 1, 0, 1, 0, 1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int32(expected)
        },
        "bigint_col" => {
            let expected = vec![0, 10, 0, 10, 0, 10, 0, 10];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Int64(expected)
        },
        "float_col" => {
            let expected = vec![0.0, 1.1, 0.0, 1.1, 0.0, 1.1, 0.0, 1.1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Float(expected)
        },
        "double_col" => {
            let expected = vec![0.0, 10.1, 0.0, 10.1, 0.0, 10.1, 0.0, 10.1];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Double(expected)
        },
        "date_string_col" => {
            let expected = vec![
                vec![48, 51, 47, 48, 49, 47, 48, 57],
                vec![48, 51, 47, 48, 49, 47, 48, 57],
                vec![48, 52, 47, 48, 49, 47, 48, 57],
                vec![48, 52, 47, 48, 49, 47, 48, 57],
                vec![48, 50, 47, 48, 49, 47, 48, 57],
                vec![48, 50, 47, 48, 49, 47, 48, 57],
                vec![48, 49, 47, 48, 49, 47, 48, 57],
                vec![48, 49, 47, 48, 49, 47, 48, 57],
            ];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Binary(expected)
        },
        "string_col" => {
            let expected = vec![
                vec![48],
                vec![49],
                vec![48],
                vec![49],
                vec![48],
                vec![49],
                vec![48],
                vec![49],
            ];
            let expected = expected.into_iter().map(Some).collect::<Vec<_>>();
            Array::Binary(expected)
        },
        "timestamp_col" => {
            todo!()
        },
        _ => unreachable!(),
    }
}

pub fn alltypes_statistics(column: &str) -> Statistics {
    match column {
        "id" => PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(7),
        }
        .into(),
        "id-short-array" => PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(4),
            max_value: Some(4),
        }
        .into(),
        "bool_col" => BooleanStatistics {
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(false),
            max_value: Some(true),
        }
        .into(),
        "tinyint_col" | "smallint_col" | "int_col" => PrimitiveStatistics::<i32> {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::Int32),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(1),
        }
        .into(),
        "bigint_col" => PrimitiveStatistics::<i64> {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::Int64),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0),
            max_value: Some(10),
        }
        .into(),
        "float_col" => PrimitiveStatistics::<f32> {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::Float),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0.0),
            max_value: Some(1.1),
        }
        .into(),
        "double_col" => PrimitiveStatistics::<f64> {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::Double),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(0.0),
            max_value: Some(10.1),
        }
        .into(),
        "date_string_col" => BinaryStatistics {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::ByteArray),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(vec![48, 49, 47, 48, 49, 47, 48, 57]),
            max_value: Some(vec![48, 52, 47, 48, 49, 47, 48, 57]),
        }
        .into(),
        "string_col" => BinaryStatistics {
            primitive_type: PrimitiveType::from_physical("col".into(), PhysicalType::ByteArray),
            null_count: Some(0),
            distinct_count: None,
            min_value: Some(vec![48]),
            max_value: Some(vec![49]),
        }
        .into(),
        "timestamp_col" => {
            todo!()
        },
        _ => unreachable!(),
    }
}

#[test]
fn test_vstack_empty_3220() -> PolarsResult<()> {
    let df1 = df! {
        "a" => ["1", "2"],
        "b" => [1, 2]
    }?;
    let empty_df = df1.head(Some(0));
    let mut stacked = df1.clone();
    stacked.vstack_mut(&empty_df)?;
    stacked.vstack_mut(&df1)?;
    let mut buf = Cursor::new(Vec::new());
    ParquetWriter::new(&mut buf).finish(&mut stacked)?;
    let read_df = ParquetReader::new(buf).finish()?;
    assert!(stacked.equals(&read_df));
    Ok(())
}

#[test]
fn test_ext_store_sink_and_scan_parquet() -> PolarsResult<()> {
    use std::num::NonZeroUsize;
    use std::sync::{Arc, Mutex};

    use object_store::ObjectStore;
    use object_store::memory::InMemory;
    use polars::prelude::*;
    use polars_core::runtime::ASYNC;
    use polars_io::cloud::cloud_writer::{CloudWriter, CloudWriterIoTraitWrap};
    use polars_io::cloud::{
        CloudConfig, CloudOptions, ExtObjectStoreBuilder, build_object_store,
        deregister_object_store_builder, register_object_store_builder,
    };
    use polars_io::prelude::ParquetWriter;
    use polars_io::utils::file::WritableTrait;
    use polars_utils::pl_path::PlRefPath;

    struct MemoryBuilder {
        store: Arc<InMemory>,
        received_options: Arc<Mutex<Vec<(String, String)>>>,
    }

    impl ExtObjectStoreBuilder for MemoryBuilder {
        fn build(
            &self,
            _url: &PlRefPath,
            options: Option<&CloudOptions>,
        ) -> PolarsResult<Arc<dyn ObjectStore + Send + Sync>> {
            if let Some(CloudOptions {
                config: Some(CloudConfig::Ext { options: ext_opts }),
                ..
            }) = options
            {
                *self.received_options.lock().unwrap() = ext_opts.clone();
            }
            Ok(self.store.clone())
        }
    }

    let received_options = Arc::new(Mutex::new(vec![]));
    let store = Arc::new(InMemory::new());
    let output_path = "pl-mem://host/data/output.parquet";

    let mut storage_options = CloudOptions::default();
    storage_options.config = Some(CloudConfig::Ext {
        options: vec![("user".to_string(), "hadoop".to_string())],
    });

    polars_utils::pl_path::_allow_ext_scheme("pl-mem")?;
    register_object_store_builder(
        "pl-mem",
        Arc::new(MemoryBuilder {
            store: store.clone(),
            received_options: received_options.clone(),
        }),
    )
    .unwrap();

    let (cloud_location, polars_store) = ASYNC.block_in_place_on(async {
        build_object_store(PlRefPath::new(output_path), Some(&storage_options), false)
            .await
            .unwrap()
    });

    let obj_path = object_store::path::Path::parse(&cloud_location.prefix).unwrap();

    let mut wrap = CloudWriterIoTraitWrap::from(CloudWriter::new(
        polars_store,
        obj_path,
        NonZeroUsize::new(8 * 1024 * 1024),
        NonZeroUsize::new(1).unwrap(),
        None,
    ));

    let mut df = df![
        "a" => [1i32, 2, 3],
        "b" => ["x", "y", "z"],
    ]?;

    // Sink
    ParquetWriter::new(&mut wrap).finish(&mut df)?;
    wrap.close()?;

    // Assert options were received
    let opts = received_options.lock().unwrap().clone();
    assert_eq!(opts.len(), 1);
    assert_eq!(opts[0], ("user".to_string(), "hadoop".to_string()));

    // Scan
    let result = LazyFrame::scan_parquet(output_path.into(), Default::default())?.collect()?;

    assert_eq!(result.shape(), (3, 2));
    assert_eq!(result.column("a")?.i32()?.cont_slice()?, &[1, 2, 3]);
    assert_eq!(
        result.column("b")?.str()?.iter().collect::<Vec<_>>(),
        vec![Some("x"), Some("y"), Some("z")]
    );

    deregister_object_store_builder("pl-mem");
    polars_utils::pl_path::_disallow_ext_scheme("pl-mem");

    Ok(())
}

#[test]
#[cfg(all(feature = "lazy", feature = "parquet"))]
fn test_empty_parquet_scan_group_by_join_29732() -> PolarsResult<()> {
    // An empty parquet scan used to panic during cardinality estimation: `IR::Scan`
    // did not floor its row count at `MIN_CARDINALITY`, so a downstream `group_by`
    // over an empty file called `0.0_f64.clamp(1.0, 0.0)`, which panics because
    // `min > max`.
    // https://github.com/pola-rs/polars/issues/29732
    use polars_io::prelude::ParquetWriter;

    // An empty frame with a single Int64 column.
    let mut empty = DataFrame::new(
        0,
        vec![Column::new(
            "k".into(),
            Series::new("k".into(), Vec::<i64>::new()),
        )],
    )?;

    let path = std::env::temp_dir().join(format!("polars_29732_{}.parquet", std::process::id()));
    let f = std::fs::File::create(&path)?;
    ParquetWriter::new(f).finish(&mut empty)?;

    let grouped = LazyFrame::scan_parquet(path.to_str().unwrap().into(), Default::default())?
        .group_by([col("k")])
        .agg([len()]);

    let out = df!["k" => [1i64]]?
        .lazy()
        .join(
            grouped,
            [col("k")],
            [col("k")],
            JoinArgs::new(JoinType::Left),
        )?
        .collect();

    let _ = std::fs::remove_file(&path);

    // The left side has one row; the right (group-by over an empty file) has no
    // groups, so `len` must be null rather than the query panicking.
    let out = out?;
    assert_eq!(out.shape(), (1, 2));
    assert_eq!(out.column("k")?.i64()?.cont_slice()?, &[1]);
    assert!(out.column("len").is_ok_and(|s| s.is_null().any()));
    Ok(())
}
