use glob::glob;
use polars::prelude::*;
use std::env;
use std::fs::{canonicalize, File};
use std::io::Write;
use std::process::exit;
use std::time::Instant;

fn read_df(f: &File) -> DataFrame {
    let mut df = CsvReader::new(f)
        .infer_schema(Some(100))
        .has_header(true)
        .finish()
        .expect("dataframe");

    // for groupby we need to cast a column to a string
    if let Ok(s) = df.column("str") {
        let s = s
            .i64()
            .expect("i64")
            .into_iter()
            .map(|v| v.map(|v| format!("{}", v)));
        let s: Series = Series::new("str", &s.collect::<Vec<_>>());
        df.replace("str", s).expect("replaced");
    }
    df
}

fn bench_groupby() {
    let paths = glob("../data/1*.csv")
        .expect("valid glob")
        .map(|v| v.expect("path"));
    let mut paths = paths.collect::<Vec<_>>();
    paths.sort();

    let mut wrt_file = File::create("../data/rust_bench.txt").expect("file");
    let mut wrt_file_str = File::create("../data/rust_bench_str.txt").expect("file");

    for p in &paths {
        let f = File::open(p).expect("a csv file");
        let df = read_df(&f);

        let now = Instant::now();
        let sum = df.groupby("groups").expect("gb").select("values").sum();
        let duration = now.elapsed().as_micros();

        let now = Instant::now();
        let sum_str = df.groupby("str").expect("gb").select("values").sum();
        let duration_str = now.elapsed().as_micros();
        println!("{:?}", (sum, sum_str));
        println!("{:?}", (p, duration));
        wrt_file
            .write(&format!("{}\n", duration).as_bytes())
            .expect("write to file");
        wrt_file_str
            .write(&format!("{}\n", duration_str).as_bytes())
            .expect("write to file");
    }
}

fn bench_join() {
    let f = File::open("../data/join_left_80000.csv").expect("file");
    let left = read_df(&f);
    let f = File::open("../data/join_right_80000.csv").expect("file");
    let right = read_df(&f);
    let mut wrt_file = File::create("../data/rust_bench_join.txt").expect("file");

    let mut mean = 0.0;
    for _ in 0..10 {
        let now = Instant::now();
        let _joined = left
            .inner_join(&right, "key", "key")
            .expect("could not join");
        let duration = now.elapsed().as_micros();
        mean += duration as f32
    }
    mean /= 10.;
    println!("inner join: {} μs", mean);
    writeln!(wrt_file, "{}", mean).expect("could not write");

    let mut mean = 0.0;
    for _ in 0..10 {
        let now = Instant::now();
        let _joined = left
            .left_join(&right, "key", "key")
            .expect("could not join");
        let duration = now.elapsed().as_micros();
        mean += duration as f32
    }
    mean /= 10.;
    println!("left join: {} μs", mean);
    writeln!(wrt_file, "{}", mean).expect("could not write");

    let mut mean = 0.0;
    for _ in 0..10 {
        let now = Instant::now();
        let _joined = left
            .outer_join(&right, "key", "key")
            .expect("could not join");
        let duration = now.elapsed().as_micros();
        mean += duration as f32
    }
    mean /= 10.;
    println!("outer join: {} μs", mean);
    writeln!(wrt_file, "{}", mean).expect("could not write");
}

fn print_cli() {
    println!(
        "
cargo run [args]

args:
        groupby
        join
        "
    );
}

fn main() {
    let args = env::args().collect::<Vec<_>>();
    if args.len() == 1 {
        print_cli();
        exit(0)
    }
    println!("Current directory: {:?}", canonicalize("."));
    match &args[1][..] {
        "groupby" => bench_groupby(),
        "join" => bench_join(),
        other => {
            println!("got {}. expected:", other);
            print_cli();
            exit(1)
        }
    }
}
