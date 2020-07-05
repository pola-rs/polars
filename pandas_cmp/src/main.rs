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
    let s = df
        .f_select_mut("str")
        .i64()
        .expect("i64")
        .into_iter()
        .map(|v| v.map(|v| format!("{}", v)));
    let s: Series = Series::new("str", &s.collect::<Vec<_>>());
    df.replace("str", s).expect("replaced");
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
    let f = File::open("../data/1000.csv").expect("file");
    let df = read_df(&f);
    let size = 500;
    let a = df.slice(0, size).expect("sliced df");
    let b = df.slice(size, size).expect("sliced df");
    let now = Instant::now();
    let joined = a.inner_join(&b, "groups", "groups").expect("join");
    let duration = now.elapsed().as_micros();
    println!("duration: {} μs", duration);
    println!("{:?}", joined);
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
