#![feature(test)]
extern crate test;
use polars::prelude::*;
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
