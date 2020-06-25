use glob::glob;
use polars::prelude::*;
use std::fs::{canonicalize, File};
use std::io::Write;
use std::time::Instant;

fn main() {
    println!("Current directory: {:?}", canonicalize("."));
    let paths = glob("../data/1*.csv")
        .expect("valid glob")
        .map(|v| v.expect("path"));
    let mut paths = paths.collect::<Vec<_>>();
    paths.sort();

    let mut wrt_file = File::create("../data/rust_bench.txt").expect("file");

    for p in &paths {
        let f = File::open(p).expect("a csv file");
        let df = CsvReader::new(f)
            .infer_schema(Some(100))
            .has_header(true)
            .finish()
            .expect("dataframe");

        let now = Instant::now();
        let sum = df.groupby("groups").expect("gb").select("values").sum();
        let duration = now.elapsed().as_micros();
        println!("{:?}", sum);
        println!("{:?}", (p, duration));
        wrt_file
            .write(&format!("{}\n", duration).as_bytes())
            .expect("write to file");
    }
}
