#![feature(test)]
extern crate test;
use polars::prelude::*;
use std::iter;
use test::Bencher;

#[bench]
fn bench_std_iter(b: &mut Bencher) {
    let v: Vec<u32> = (0..1000).collect();
    let mut sum = 0;
    b.iter(|| sum = v.iter().sum::<u32>());
    println!("{}", sum)
}

#[bench]
fn bench_warmup(b: &mut Bencher) {
    let s: Series = (0u32..1000).collect();
    b.iter(|| {
        s.u32().unwrap().into_iter();
    });
}

#[bench]
fn bench_num_iter(b: &mut Bencher) {
    let s: Series = (0u32..1000).collect();
    let mut sum = 0;
    b.iter(|| {
        sum = s
            .u32()
            .unwrap()
            .into_iter()
            .map(|opt| opt.unwrap())
            .sum::<u32>()
    });
    println!("{}", sum)
}

#[bench]
fn bench_num_2_chunks(b: &mut Bencher) {
    let mut s: Series = (0u32..500).collect();
    let s2: Series = (500u32..1000).collect();
    s.append(&s2).unwrap();
    let mut sum = 0;
    b.iter(|| {
        sum = s
            .u32()
            .unwrap()
            .into_iter()
            .map(|opt| opt.unwrap())
            .sum::<u32>()
    });
    println!("{}", sum)
}

#[bench]
fn bench_join_2_frames(b: &mut Bencher) {
    let s1: Series = Series::new("id", (0u32..10000).collect::<Vec<u32>>());
    let s2: Series = Series::new("id", (0u32..10000).collect::<Vec<u32>>());

    let df1 = DataFrame::new(vec![s1]).unwrap();

    let df2 = DataFrame::new(vec![s2]).unwrap();

    let mut sum = 0;

    b.iter(|| {
        let df3 = df1.inner_join(&df2, "id", "id").unwrap();
        sum += df3.shape().1;
    });

    println!("{}", sum)
}

#[bench]
fn bench_group_by(b: &mut Bencher) {
    let s1: Series = Series::new("item", (0u32..10000).collect::<Vec<u32>>());
    let s2: Series = Series::new("group", iter::repeat(0).take(10000).collect::<Vec<u32>>());

    let df1 = DataFrame::new(vec![s1, s2]).unwrap();

    b.iter(|| {
        df1.groupby("group").unwrap().select("item").sum().unwrap();
    });
}
