//! Empirical rank-error study for the approximate quantile sketches.
//!
//! Prints one CSV row per (method, error, n, threads, split, quantile) with the
//! distribution of the observed rank error over `--trials` independent runs.

use std::env;
use std::sync::Mutex;

use polars_compute::approx_quantile::{ApproxQuantileMethod, Sketch};
use rand::rngs::SmallRng;
use rand::{RngExt, SeedableRng};

#[derive(Clone, Copy, Debug, PartialEq)]
enum Split {
    /// All parts the same size.
    Even,
    /// Dirichlet(1) sizes: moderately uneven.
    Dirichlet,
    /// Very skewed sizes, many parts end up (near) empty.
    Skewed,
    /// One part holds almost everything, the rest get 1..=16 items.
    OneBig,
}

impl Split {
    fn name(self) -> &'static str {
        match self {
            Split::Even => "even",
            Split::Dirichlet => "dirichlet",
            Split::Skewed => "skewed",
            Split::OneBig => "onebig",
        }
    }

    /// Sizes of the `parts` streams, summing to `n`.
    fn sizes(self, n: usize, parts: usize, rng: &mut SmallRng) -> Vec<usize> {
        if parts == 1 {
            return vec![n];
        }
        let mut sizes = match self {
            Split::Even => {
                let base = n / parts;
                let mut sizes = vec![base; parts];
                for s in sizes.iter_mut().take(n - base * parts) {
                    *s += 1;
                }
                return sizes;
            },
            Split::OneBig => {
                let mut sizes = vec![0usize; parts];
                let mut used = 0;
                for s in sizes.iter_mut().skip(1) {
                    *s = rng.random_range(1..=16).min(n.saturating_sub(used + 1));
                    used += *s;
                }
                sizes[0] = n - used;
                return sizes;
            },
            Split::Dirichlet => {
                // Dirichlet(1) via normalized Exp(1) weights.
                let w: Vec<f64> = (0..parts)
                    .map(|_| -f64::ln(1.0 - rng.random::<f64>()))
                    .collect();
                w
            },
            Split::Skewed => {
                // Heavily skewed: u^(1/alpha) with a small alpha.
                const ALPHA: f64 = 0.15;
                (0..parts)
                    .map(|_| f64::powf(rng.random::<f64>().max(1e-12), 1.0 / ALPHA))
                    .collect()
            },
        };
        let total: f64 = sizes.iter().sum();
        let mut out: Vec<usize> = sizes
            .iter_mut()
            .map(|w| (*w / total * n as f64) as usize)
            .collect();
        let assigned: usize = out.iter().sum();
        out[0] += n - assigned;
        out
    }
}

/// Scratch buffers reused across the trials of one worker thread.
#[derive(Default)]
struct Scratch {
    data: Vec<f64>,
    bucket_of: Vec<u32>,
    offsets: Vec<usize>,
}

/// Fill `scratch.data` with a uniform random permutation of `0..n`, so a value
/// equals its own 0-based rank.
///
/// A plain Fisher-Yates over `n` items is dominated by cache misses, so bucket
/// the values first (sequential writes into `B` streams) and shuffle each bucket
/// in cache. Concatenating independently shuffled buckets of a uniform random
/// bucket assignment is again a uniform random permutation.
fn permutation(n: usize, rng: &mut SmallRng, scratch: &mut Scratch) {
    const BUCKET_SIZE: usize = 2048;
    let buckets = usize::max(1, n / BUCKET_SIZE);

    scratch.bucket_of.clear();
    scratch.bucket_of.reserve(n);
    scratch.offsets.clear();
    scratch.offsets.resize(buckets + 1, 0);
    for _ in 0..n {
        let b = rng.random_range(0..buckets);
        scratch.bucket_of.push(b as u32);
        scratch.offsets[b + 1] += 1;
    }
    for i in 0..buckets {
        scratch.offsets[i + 1] += scratch.offsets[i];
    }

    let data = &mut scratch.data;
    data.clear();
    data.resize(n, 0.0);
    let mut cursor = scratch.offsets.clone();
    for (v, b) in scratch.bucket_of.iter().enumerate() {
        let slot = &mut cursor[*b as usize];
        data[*slot] = v as f64;
        *slot += 1;
    }
    for w in scratch.offsets.windows(2) {
        let bucket = &mut data[w[0]..w[1]];
        for i in (1..bucket.len()).rev() {
            bucket.swap(i, rng.random_range(0..=i));
        }
    }
}

/// Build one sketch per part, merge them, and return the finalized sketch.
fn run_trial(
    method: &ApproxQuantileMethod,
    error: f64,
    data: &[f64],
    sizes: &[usize],
) -> Sketch<f64> {
    let mut sketches = Vec::with_capacity(sizes.len());
    let mut offset = 0;
    for &size in sizes {
        let mut sketch = Sketch::new(method, error);
        for v in &data[offset..offset + size] {
            sketch.update(v);
        }
        sketch.finalize();
        offset += size;
        sketches.push(sketch);
    }
    // Merge like a reduction tree, as the engines do.
    while sketches.len() > 1 {
        let mut next = Vec::with_capacity(sketches.len().div_ceil(2));
        let mut it = sketches.into_iter();
        while let Some(mut a) = it.next() {
            if let Some(b) = it.next() {
                a.merge(b);
            }
            next.push(a);
        }
        sketches = next;
    }
    sketches.pop().unwrap()
}

/// Denominator that turns a rank error into the error notion the method
/// promises: `n` for KLL (uniform) and the distance to the nearer end of the
/// stream for the relative-error sketches.
fn rel_denom(method_name: &str, q: f64, n: usize) -> f64 {
    let r = (q * (n - 1) as f64).round() + 1.0;
    let lo = f64::max(r, 1.0);
    let hi = f64::max(n as f64 - r + 1.0, 1.0);
    match method_name {
        "req_lra" => lo,
        "req_hra" => hi,
        _ => f64::min(lo, hi),
    }
}

fn percentile(sorted: &[f64], p: f64) -> f64 {
    if sorted.is_empty() {
        return f64::NAN;
    }
    let idx = (p * (sorted.len() - 1) as f64).round() as usize;
    sorted[idx]
}

struct Config {
    method: ApproxQuantileMethod,
    method_name: &'static str,
    error: f64,
    n: usize,
    parts: usize,
    split: Split,
    quantiles: Vec<f64>,
    trials: usize,
    /// Arrival order of the values: "random", "sorted" or "reverse".
    order: &'static str,
    /// Emit only the per-trial worst error over the whole quantile grid.
    grid_only: bool,
}

fn method_by_name(name: &str) -> (ApproxQuantileMethod, &'static str) {
    match name {
        "kll" => (ApproxQuantileMethod::KLL, "kll"),
        "req_lra" => (ApproxQuantileMethod::ReqSketch { hra: false }, "req_lra"),
        "req_hra" => (ApproxQuantileMethod::ReqSketch { hra: true }, "req_hra"),
        "double_req" => (ApproxQuantileMethod::DoubleReqSketch, "double_req"),
        other => panic!("unknown method: {other}"),
    }
}

fn run_config(cfg: &Config, threads: usize) {
    let nq = cfg.quantiles.len();
    // errors[q] holds the signed rank error of every trial.
    #[allow(clippy::type_complexity)]
    let shared: Mutex<(Vec<Vec<f64>>, Vec<f64>, Vec<f64>)> = Mutex::new((
        vec![Vec::with_capacity(cfg.trials); nq],
        Vec::with_capacity(cfg.trials),
        Vec::with_capacity(cfg.trials),
    ));
    let next_trial = std::sync::atomic::AtomicUsize::new(0);

    std::thread::scope(|scope| {
        for _ in 0..threads {
            scope.spawn(|| {
                let mut local: Vec<Vec<f64>> = vec![Vec::new(); nq];
                let mut local_max_abs: Vec<f64> = Vec::new();
                let mut local_max_rel: Vec<f64> = Vec::new();
                let denoms: Vec<f64> = cfg
                    .quantiles
                    .iter()
                    .map(|q| rel_denom(cfg.method_name, *q, cfg.n))
                    .collect();
                let mut scratch = Scratch::default();
                loop {
                    let trial = next_trial.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                    if trial >= cfg.trials {
                        break;
                    }
                    let seed = (trial as u64)
                        .wrapping_mul(0x9E3779B97F4A7C15)
                        .wrapping_add(cfg.n as u64)
                        .wrapping_add((cfg.error * 1e9) as u64)
                        .wrapping_add(cfg.parts as u64);
                    let mut rng = SmallRng::seed_from_u64(seed);
                    let sizes = cfg.split.sizes(cfg.n, cfg.parts, &mut rng);
                    permutation(cfg.n, &mut rng, &mut scratch);
                    match cfg.order {
                        "sorted" => scratch.data.sort_by(f64::total_cmp),
                        "reverse" => scratch.data.sort_by(|a, b| f64::total_cmp(b, a)),
                        _ => {},
                    }
                    let data = &scratch.data;
                    let sketch = run_trial(&cfg.method, cfg.error, data, &sizes);
                    let (mut max_abs, mut max_rel) = (0.0f64, 0.0f64);
                    for (i, &q) in cfg.quantiles.iter().enumerate() {
                        let got = *sketch.estimate_quantile(q).unwrap();
                        let want = (q * (cfg.n - 1) as f64).round();
                        let err = got - want;
                        if !cfg.grid_only {
                            local[i].push(err);
                        }
                        max_abs = max_abs.max(err.abs());
                        max_rel = max_rel.max(err.abs() / denoms[i]);
                    }
                    local_max_abs.push(max_abs);
                    local_max_rel.push(max_rel);
                }
                let mut guard = shared.lock().unwrap();
                for (dst, src) in guard.0.iter_mut().zip(local) {
                    dst.extend(src);
                }
                guard.1.extend(local_max_abs);
                guard.2.extend(local_max_rel);
            });
        }
    });

    let (errors, max_abs_per_trial, max_rel_per_trial) = shared.into_inner().unwrap();

    // The per-trial worst over the whole quantile grid: how far off the *worst*
    // answered quantile of one sketch is.
    {
        let n = cfg.n as f64;
        let mut abs = max_abs_per_trial.clone();
        let mut rel = max_rel_per_trial.clone();
        abs.sort_by(f64::total_cmp);
        rel.sort_by(f64::total_cmp);
        let count = abs.len() as f64;
        let mean: f64 = abs.iter().sum::<f64>() / count;
        let std_abs = f64::sqrt(abs.iter().map(|e| e * e).sum::<f64>() / count);
        let std_rel = f64::sqrt(rel.iter().map(|e| e * e).sum::<f64>() / count);
        println!(
            "{},{},{},{},{},{},grid,{},{:.6},{:.8},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8},{:.6}",
            cfg.method_name,
            cfg.error,
            cfg.n,
            cfg.parts,
            cfg.split.name(),
            cfg.order,
            abs.len(),
            mean / n,
            std_abs / n,
            percentile(&abs, 0.5) / n,
            percentile(&abs, 0.9973) / n,
            abs[abs.len() - 1] / n,
            std_rel,
            percentile(&rel, 0.5),
            percentile(&rel, 0.9973),
            rel[rel.len() - 1],
            f64::NAN,
        );
    }
    if cfg.grid_only {
        return;
    }

    for (i, &q) in cfg.quantiles.iter().enumerate() {
        let n = cfg.n as f64;
        let rel_denom = rel_denom(cfg.method_name, q, cfg.n);
        let mut abs: Vec<f64> = errors[i].iter().map(|e| e.abs()).collect();
        let mut rel: Vec<f64> = errors[i].iter().map(|e| e.abs() / rel_denom).collect();
        abs.sort_by(f64::total_cmp);
        rel.sort_by(f64::total_cmp);
        let count = errors[i].len() as f64;
        let mean: f64 = errors[i].iter().sum::<f64>() / count;
        let std_abs = f64::sqrt(errors[i].iter().map(|e| e * e).sum::<f64>() / count);
        let std_rel = f64::sqrt(
            errors[i]
                .iter()
                .map(|e| (e / rel_denom) * (e / rel_denom))
                .sum::<f64>()
                / count,
        );
        println!(
            "{},{},{},{},{},{},{},{},{:.6},{:.8},{:.6},{:.6},{:.6},{:.6},{:.6},{:.6},{:.8},{:.6}",
            cfg.method_name,
            cfg.error,
            cfg.n,
            cfg.parts,
            cfg.split.name(),
            cfg.order,
            q,
            errors[i].len(),
            mean / n,
            std_abs / n,
            percentile(&abs, 0.5) / n,
            percentile(&abs, 0.9973) / n,
            abs[abs.len() - 1] / n,
            std_rel,
            percentile(&rel, 0.5),
            percentile(&rel, 0.9973),
            rel[rel.len() - 1],
            rel_denom / n,
        );
    }
}

fn parse_list<T: std::str::FromStr>(s: &str) -> Vec<T>
where
    T::Err: std::fmt::Debug,
{
    s.split(',').map(|x| x.trim().parse().unwrap()).collect()
}

fn main() {
    let mut methods = vec!["kll".to_string()];
    let mut errors = vec![0.01f64];
    let mut ns = vec![1_000_000usize];
    let mut parts = vec![1usize, 8, 64];
    let mut splits = vec![Split::Even, Split::Dirichlet, Split::Skewed, Split::OneBig];
    let mut quantiles: Vec<f64> = vec![];
    let mut trials = 1000usize;
    let mut order = "random".to_string();
    let mut qgrid = 0usize;
    let mut threads = 24usize;

    let args: Vec<String> = env::args().skip(1).collect();
    let mut i = 0;
    while i < args.len() {
        let val = args.get(i + 1).expect("missing value").as_str();
        match args[i].as_str() {
            "--methods" => methods = val.split(',').map(|s| s.to_string()).collect(),
            "--errors" => errors = parse_list(val),
            "--n" => ns = parse_list(val),
            "--parts" => parts = parse_list(val),
            "--splits" => {
                splits = val
                    .split(',')
                    .map(|s| match s {
                        "even" => Split::Even,
                        "dirichlet" => Split::Dirichlet,
                        "skewed" => Split::Skewed,
                        "onebig" => Split::OneBig,
                        o => panic!("unknown split {o}"),
                    })
                    .collect()
            },
            "--quantiles" => quantiles = parse_list(val),
            "--trials" => trials = val.parse().unwrap(),
            "--order" => order = val.to_string(),
            "--qgrid" => qgrid = val.parse().unwrap(),
            "--threads" => threads = val.parse().unwrap(),
            o => panic!("unknown flag {o}"),
        }
        i += 2;
    }

    println!(
        "method,error,n,parts,split,order,q,trials,mean_abs_frac,std_abs_frac,p50_abs_frac,p9973_abs_frac,max_abs_frac,std_rel,p50_rel,p9973_rel,max_rel,rel_denom_frac"
    );
    for method_name in &methods {
        let (method, method_name) = method_by_name(method_name);
        let qs = if qgrid > 0 {
            // Log-spaced towards the tail the method is accurate in, plus the
            // exact ends of the range.
            (0..=qgrid)
                .map(|i| {
                    let t = i as f64 / qgrid as f64;
                    match method_name {
                        "kll" => t,
                        "req_hra" => 1.0 - 0.5 * f64::powf(1e-6, 1.0 - t),
                        _ => 0.5 * f64::powf(1e-6, 1.0 - t),
                    }
                })
                .collect()
        } else if !quantiles.is_empty() {
            quantiles.clone()
        } else {
            match method_name {
                "kll" => vec![0.0, 0.01, 0.05, 0.1, 0.25, 0.5, 0.75, 0.9, 0.95, 0.99, 1.0],
                "req_lra" => vec![0.0001, 0.001, 0.01, 0.05, 0.1, 0.25, 0.5],
                "req_hra" => vec![0.5, 0.75, 0.9, 0.95, 0.99, 0.999, 0.9999],
                _ => vec![0.0001, 0.001, 0.01, 0.1, 0.5, 0.9, 0.99, 0.999, 0.9999],
            }
        };
        for &error in &errors {
            for &n in &ns {
                for &p in &parts {
                    for &split in &splits {
                        if p == 1 && split != Split::Even {
                            continue;
                        }
                        let order: &'static str = match order.as_str() {
                            "random" => "random",
                            "sorted" => "sorted",
                            "reverse" => "reverse",
                            o => panic!("unknown order {o}"),
                        };
                        let cfg = Config {
                            method: method.clone(),
                            method_name,
                            error,
                            n,
                            parts: p,
                            split,
                            quantiles: qs.clone(),
                            trials,
                            order,
                            grid_only: qgrid > 0,
                        };
                        run_config(&cfg, threads);
                    }
                }
            }
        }
    }
}
