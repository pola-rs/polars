/// Since the HybridRle decoder is very widely used within the Parquet reader and the code is quite
/// complex to facilitate performance. We create this small fuzzer
use std::collections::VecDeque;

use rand::Rng;

use super::*;

fn run_iteration(
    bs: &[u32],
    collects: impl Iterator<Item = usize>,
    encoded: &mut Vec<u8>,
    decoded: &mut Vec<u32>,
    num_bits: u32,
) -> ParquetResult<()> {
    encoded.clear();
    decoded.clear();

    encoder::encode(encoded, bs.iter().copied(), num_bits).unwrap();

    let mut decoder = HybridRleDecoder::new(&encoded[..], num_bits, bs.len());

    for c in collects {
        decoder.collect_n_into(decoded, c)?;
    }

    Ok(())
}

/// Minimizes a failing case
fn minimize_failing_case(
    bs: &mut Vec<u32>,
    collects: &mut VecDeque<usize>,
    encoded: &mut Vec<u8>,
    decoded: &mut Vec<u32>,
    num_bits: u32,
) -> ParquetResult<()> {
    loop {
        let initial_bs_len = bs.len();
        let initial_collects_len = collects.len();

        run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

        assert_ne!(&bs, &decoded);

        while collects.len() > 2 {
            let last = collects.pop_back().unwrap();

            *collects.back_mut().unwrap() += last;

            run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

            if bs == decoded {
                *collects.back_mut().unwrap() -= last;
                collects.push_back(last);
                break;
            }
        }

        while collects.len() > 2 {
            let first = collects.pop_front().unwrap();

            *collects.front_mut().unwrap() += first;

            run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

            if bs == decoded {
                *collects.front_mut().unwrap() -= first;
                collects.push_front(first);
                break;
            }
        }

        while bs.len() > 1 {
            let last = bs.pop().unwrap();
            *collects.back_mut().unwrap() -= 1;

            run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

            if bs == decoded {
                bs.push(last);
                *collects.back_mut().unwrap() += 1;
                break;
            }

            if *collects.back().unwrap() == 0 {
                collects.pop_back().unwrap();

                run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                if bs == decoded {
                    collects.push_back(0);
                    break;
                }
            }
        }

        while bs.len() > 1 {
            let last = bs.pop().unwrap();
            *collects.front_mut().unwrap() -= 1;

            run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

            if bs == decoded {
                bs.push(last);
                *collects.front_mut().unwrap() += 1;
                break;
            }

            if *collects.front().unwrap() == 0 {
                collects.pop_front().unwrap();

                run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                if bs == decoded {
                    collects.push_front(0);
                    break;
                }
            }
        }

        while bs.len() > 1 {
            let first = bs.remove(0);
            *collects.back_mut().unwrap() -= 1;

            run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

            if bs == decoded {
                bs.insert(0, first);
                *collects.back_mut().unwrap() += 1;
                break;
            }

            if *collects.back().unwrap() == 0 {
                collects.pop_back().unwrap();

                run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                if bs == decoded {
                    collects.push_back(0);
                    break;
                }
            }
        }

        while bs.len() > 1 {
            let first = bs.remove(0);
            *collects.front_mut().unwrap() -= 1;

            run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

            if bs == decoded {
                bs.insert(0, first);
                *collects.front_mut().unwrap() += 1;
                break;
            }

            if *collects.front().unwrap() == 0 {
                collects.pop_front().unwrap();

                run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                if bs == decoded {
                    collects.push_front(0);
                    break;
                }
            }
        }

        let mut start_offset = collects[0];
        for i in 1..collects.len() - 1 {
            loop {
                let start_length = collects[i];

                while collects[i] > 0 {
                    collects[i] -= 1;
                    let item = bs.remove(start_offset);

                    run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                    if bs == decoded {
                        bs.insert(start_offset, item);
                        collects[i] += 1;
                        break;
                    }

                    if collects[i] == 0 {
                        collects.remove(i);

                        run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                        if bs == decoded {
                            collects.insert(i, 0);
                            break;
                        }
                    }
                }

                while collects[i] > 0 {
                    collects[i] -= 1;
                    let end_offset = start_offset + collects[i] - 1;
                    let item = bs.remove(end_offset);

                    run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                    if bs == decoded {
                        bs.insert(end_offset, item);
                        collects[i] += 1;
                        break;
                    }

                    if collects[i] == 0 {
                        collects.remove(i);

                        run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

                        if bs == decoded {
                            collects.insert(i, 0);
                            break;
                        }
                    }
                }

                if collects[i] == start_length {
                    break;
                }
            }

            start_offset += collects[i];
        }

        let now_bs_len = bs.len();
        let now_collects_len = collects.len();

        if initial_bs_len == now_bs_len && initial_collects_len == now_collects_len {
            break;
        }
    }

    run_iteration(bs, collects.iter().copied(), encoded, decoded, num_bits)?;

    Ok(())
}

fn fuzz_loops(num_loops: usize) -> ParquetResult<()> {
    let mut rng = rand::thread_rng();

    const MAX_LENGTH: usize = 10_000;

    let mut encoded = Vec::with_capacity(1024);
    let mut decoded = Vec::with_capacity(1024);

    let mut bs = Vec::with_capacity(MAX_LENGTH);
    let mut collects: VecDeque<usize> = VecDeque::with_capacity(2000);

    for i in 0..num_loops {
        collects.clear();
        bs.clear();

        let num_bits = rng.gen_range(0..=32);
        let mask = 1u32.wrapping_shl(num_bits).wrapping_sub(1);

        let length = rng.gen_range(1..=MAX_LENGTH);

        unsafe { bs.set_len(length) };
        rng.fill(&mut bs[..]);

        let mut filled = 0;
        while filled < bs.len() {
            if rng.gen() {
                let num_repeats = rng.gen_range(0..=(bs.len() - filled));
                let value = bs[filled] & mask;
                for j in 0..num_repeats {
                    bs[filled + j] = value;
                }
                filled += num_repeats;
            } else {
                bs[filled] &= mask;
                filled += 1;
            }
        }

        if rng.gen() {
            let mut num_values = bs.len();
            while num_values > 0 {
                let n = rng.gen_range(0..=num_values);
                collects.push_back(n);
                num_values -= n;
            }
        } else {
            collects.resize(1, bs.len());
        }

        run_iteration(
            &bs,
            collects.iter().copied(),
            &mut encoded,
            &mut decoded,
            num_bits,
        )?;

        if decoded != bs {
            minimize_failing_case(&mut bs, &mut collects, &mut encoded, &mut decoded, num_bits)?;

            eprintln!("Minimized case:");
            eprintln!("Expected: {bs:?}");
            eprintln!("Found:    {decoded:?}");
            eprintln!("Collects: {collects:?}");
            eprintln!();

            panic!("Found a failing case...");
        }

        if i % 512 == 0 {
            eprintln!("{i} iterations done.");
        }
    }

    Ok(())
}

#[test]
fn small_fuzz() -> ParquetResult<()> {
    fuzz_loops(2048)
}

#[test]
#[ignore = "Large fuzz test. Too slow"]
fn large_fuzz() -> ParquetResult<()> {
    fuzz_loops(1_000_000)
}

#[test]
#[ignore = "Large fuzz test. Too slow"]
fn skip_fuzz() -> ParquetResult<()> {
    let mut rng = rand::thread_rng();

    const MAX_LENGTH: usize = 10_000;

    let mut encoded = Vec::with_capacity(10000);

    let mut bs: Vec<u32> = Vec::with_capacity(MAX_LENGTH);
    let mut skips: VecDeque<usize> = VecDeque::with_capacity(2000);

    let num_loops = 100_000;

    for _ in 0..num_loops {
        skips.clear();
        bs.clear();

        let num_bits = rng.gen_range(0..=32);
        let mask = 1u32.wrapping_shl(num_bits).wrapping_sub(1);

        let length = rng.gen_range(1..=MAX_LENGTH);

        unsafe { bs.set_len(length) };
        rng.fill(&mut bs[..]);

        let mut filled = 0;
        while filled < bs.len() {
            if rng.gen() {
                let num_repeats = rng.gen_range(0..=(bs.len() - filled));
                let value = bs[filled] & mask;
                for j in 0..num_repeats {
                    bs[filled + j] = value;
                }
                filled += num_repeats;
            } else {
                bs[filled] &= mask;
                filled += 1;
            }
        }

        let mut num_done = 0;
        while num_done < filled {
            let num_skip = rng.gen_range(1..=filled - num_done);
            num_done += num_skip;
            skips.push_back(num_skip);
        }

        encoder::encode(&mut encoded, bs.iter().copied(), num_bits).unwrap();
        let mut decoder = HybridRleDecoder::new(&encoded, num_bits, filled);

        for s in &skips {
            decoder.skip_in_place(*s).unwrap();
        }
    }

    Ok(())
}
