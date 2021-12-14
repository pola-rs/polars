use criterion::{criterion_group, criterion_main, Criterion};
use lazy_static::lazy_static;
use polars::prelude::*;
use polars_lazy::functions::pearson_corr;

lazy_static! {
    static ref DATA: DataFrame = {
        let path =
            std::env::var("CSV_SRC").expect("env var CSV_SRC pointing to the csv_file is not set");

        let mut df = CsvReader::from_path(&path)
            .expect("could not read file")
            // 1M rows
            .with_n_rows(Some(1000000))
            .finish()
            .unwrap();
        df.may_apply("id1", |s| s.cast(&DataType::Categorical))
            .unwrap();
        df.may_apply("id2", |s| s.cast(&DataType::Categorical))
            .unwrap();
        df.may_apply("id3", |s| s.cast(&DataType::Categorical))
            .unwrap();
        df
    };
}

fn q1(c: &mut Criterion) {
    c.bench_function("groupby q1", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id1")])
                .agg([col("v1").sum()])
                .collect()
                .unwrap();
        })
    });
}

fn q2(c: &mut Criterion) {
    c.bench_function("groupby q2", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id1"), col("id2")])
                .agg([col("v1").sum()])
                .collect()
                .unwrap();
        })
    });
}

fn q3(c: &mut Criterion) {
    c.bench_function("groupby q3", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id3")])
                .agg([col("v1").sum(), col("v3").mean()])
                .collect()
                .unwrap();
        })
    });
}

fn q4(c: &mut Criterion) {
    c.bench_function("groupby q4", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id4")])
                .agg([col("v1").mean(), col("v2").mean(), col("v3").mean()])
                .collect()
                .unwrap();
        })
    });
}

fn q5(c: &mut Criterion) {
    c.bench_function("groupby q5", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id6")])
                .agg([col("v1").sum(), col("v2").sum(), col("v3").sum()])
                .collect()
                .unwrap();
        })
    });
}

fn q6(c: &mut Criterion) {
    c.bench_function("groupby q6", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id4"), col("id5")])
                .agg([
                    col("v3").median().alias("v3_median"),
                    col("v3").std().alias("v3_std"),
                ])
                .collect()
                .unwrap();
        })
    });
}

fn q7(c: &mut Criterion) {
    c.bench_function("groupby q7", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([col("id3")])
                .agg([col("v1").max().alias("v1"), col("v2").min().alias("v2")])
                .select([col("id3"), (col("v1") - col("v2")).alias("range_v1_v2")])
                .collect()
                .unwrap();
        })
    });
}

fn q8(c: &mut Criterion) {
    c.bench_function("groupby q8", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                // todo! accept slice of str
                .drop_nulls(Some(vec![col("v3")]))
                .sort("v3", true)
                .groupby([col("id6")])
                .agg([col("v3").head(Some(2)).alias("v3_top_2")])
                .explode(vec![col("v3_top_2")])
                .collect()
                .unwrap();
        })
    });
}

fn q9(c: &mut Criterion) {
    c.bench_function("groupby q9", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .drop_nulls(Some(vec![col("v1"), col("v2")]))
                .groupby([col("id2"), col("id4")])
                .agg([pearson_corr(col("v1"), col("v2")).alias("r2").pow(2.0)])
                .collect()
                .unwrap();
        })
    });
}

fn q10(c: &mut Criterion) {
    c.bench_function("groupby q10", |b| {
        b.iter(|| {
            DATA.clone()
                .lazy()
                .groupby([
                    col("id1"),
                    col("id2"),
                    col("id3"),
                    col("id4"),
                    col("id5"),
                    col("id6"),
                ])
                .agg([col("v3").sum().alias("v3"), col("v1").count().alias("v1")])
                .collect()
                .unwrap();
        })
    });
}

criterion_group!(name = benches;
config = Criterion::default().sample_size(100);
targets = q1, q2, q3, q4, q5, q6, q7, q8, q9, q10);
criterion_main!(benches);
