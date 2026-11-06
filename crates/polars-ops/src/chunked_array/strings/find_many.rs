use aho_corasick::{AhoCorasick, AhoCorasickBuilder, MatchKind};
use polars_arrow::array::Utf8ViewArray;
use polars_core::prelude::arity::{broadcast_try_binary_elementwise, unary_elementwise};
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

fn build_ac(
    patterns: &StringChunked,
    ascii_case_insensitive: bool,
    leftmost: bool,
) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
        .match_kind(if leftmost {
            MatchKind::LeftmostFirst
        } else {
            MatchKind::Standard
        })
        .ascii_case_insensitive(ascii_case_insensitive)
        .build(patterns.downcast_iter().flatten().flatten())
        .map_err(|e| polars_err!(ComputeError: "could not build aho corasick automaton {}", e))
}

fn build_ac_arr(
    patterns: &Utf8ViewArray,
    ascii_case_insensitive: bool,
    leftmost: bool,
) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
        .match_kind(if leftmost {
            MatchKind::LeftmostFirst
        } else {
            MatchKind::Standard
        })
        .ascii_case_insensitive(ascii_case_insensitive)
        .build(patterns.into_iter().flatten())
        .map_err(|e| polars_err!(ComputeError: "could not build aho corasick automaton {}", e))
}

pub fn contains_any(
    ca: &StringChunked,
    patterns: &ListChunked,
    ascii_case_insensitive: bool,
) -> PolarsResult<BooleanChunked> {
    polars_ensure!(
        ca.len() == patterns.len() || ca.len() == 1 || patterns.len() == 1,
        length_mismatch = "str.contains_any",
        ca.len(),
        patterns.len()
    );
    if patterns.len() != 1 {
        return broadcast_try_binary_elementwise(ca, patterns, |val, pat| {
            let (Some(val), Some(pat)) = (val, pat) else {
                return Ok(None);
            };
            let pat = pat.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
            let ac = build_ac_arr(pat, ascii_case_insensitive, false)?;
            Ok(Some(ac.find(val).is_some()))
        });
    }

    if patterns.has_nulls() {
        return Ok(BooleanChunked::full_null(ca.name().clone(), ca.len()));
    }

    let patterns = patterns.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let patterns = patterns.str()?;
    let ac = build_ac(patterns, ascii_case_insensitive, false)?;

    Ok(unary_elementwise(ca, |opt_val| {
        opt_val.map(|val| ac.find(val).is_some())
    }))
}

pub fn replace_all(
    ca: &StringChunked,
    patterns: &ListChunked,
    replace_with: &ListChunked,
    ascii_case_insensitive: bool,
    leftmost: bool,
) -> PolarsResult<StringChunked> {
    let mut length = 1;
    for (argument_idx, (argument, l)) in [
        ("self", ca.len()),
        ("patterns", patterns.len()),
        ("replace_with", replace_with.len()),
    ]
    .into_iter()
    .enumerate()
    {
        if l != 1 {
            if l != length && length != 1 {
                polars_bail!(
                    length_mismatch = "str.replace_many",
                    l,
                    length,
                    argument = argument,
                    argument_idx = argument_idx
                );
            }
            length = l;
        }
    }

    if patterns.len() != 1 || replace_with.len() != 1 {
        let broadcast = |ca: &ListChunked| {
            if ca.len() == 1 {
                ca.new_from_index(0, length)
            } else {
                ca.clone()
            }
        };
        let (patterns, replace_with) = (broadcast(patterns), broadcast(replace_with));
        let mut buf = String::new();
        let out: StringChunked = (0..length)
            .zip(patterns.amortized_iter())
            .zip(replace_with.amortized_iter())
            .map(|((i, pat), with)| {
                let val = ca.get(if ca.len() == 1 { 0 } else { i });
                let (Some(val), Some(pat), Some(with)) = (val, pat, with) else {
                    return Ok(None);
                };
                let (ac, with) = build_replacer(
                    pat.as_ref().str()?,
                    with.as_ref().str()?,
                    ascii_case_insensitive,
                    leftmost,
                )?;
                buf.clear();
                ac.replace_all_with(val, &mut buf, |m, _, dst| {
                    dst.push_str(with[m.pattern().as_usize()]);
                    true
                });
                Ok(Some(buf.clone()))
            })
            .collect::<PolarsResult<_>>()?;
        return Ok(out.with_name(ca.name().clone()));
    }

    if patterns.has_nulls() || replace_with.has_nulls() {
        return Ok(StringChunked::full_null(ca.name().clone(), ca.len()));
    }

    let patterns = patterns.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let replace_with = replace_with.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let (ac, replace_with) = build_replacer(
        patterns.str()?,
        replace_with.str()?,
        ascii_case_insensitive,
        leftmost,
    )?;

    Ok(ca.apply_into_string_amortized(|val, buf| {
        ac.replace_all_with(val, buf, |m, _, dst| {
            dst.push_str(replace_with[m.pattern().as_usize()]);
            true
        })
    }))
}

/// Builds the automaton for `patterns` and the replacement for each of them, broadcasting a single
/// replacement.
fn build_replacer<'a>(
    patterns: &StringChunked,
    replace_with: &'a StringChunked,
    ascii_case_insensitive: bool,
    leftmost: bool,
) -> PolarsResult<(AhoCorasick, Vec<&'a str>)> {
    polars_ensure!(
        replace_with.len() == 1 || replace_with.len() == patterns.len(),
        ShapeMismatch: "expected the same amount of patterns as replacement strings"
    );
    polars_ensure!(patterns.null_count() == 0 && replace_with.null_count() == 0, InvalidOperation: "'patterns'/'replace_with' should not have nulls");
    let mut replace_with: Vec<&str> = replace_with.iter().flatten().collect();
    if replace_with.len() == 1 {
        replace_with = vec![replace_with[0]; patterns.len()];
    }
    let ac = build_ac(patterns, ascii_case_insensitive, leftmost)?;
    Ok((ac, replace_with))
}

fn push_str(
    val: &str,
    builder: &mut ListStringChunkedBuilder,
    ac: &AhoCorasick,
    overlapping: bool,
) {
    if overlapping {
        let iter = ac.find_overlapping_iter(val);
        let iter = iter.map(|m| &val[m.start()..m.end()]);
        builder.append_values_iter(iter);
    } else {
        let iter = ac.find_iter(val);
        let iter = iter.map(|m| &val[m.start()..m.end()]);
        builder.append_values_iter(iter);
    }
}

pub fn extract_many(
    ca: &StringChunked,
    patterns: &ListChunked,
    ascii_case_insensitive: bool,
    overlapping: bool,
    leftmost: bool,
) -> PolarsResult<ListChunked> {
    // ensure that either overlapping == false, or overlapping == true and leftmost == false
    polars_ensure!(!overlapping | !leftmost, InvalidOperation: "can not match overlapping patterns when leftmost == True");
    match (ca.len(), patterns.len()) {
        (1, _) => match ca.get(0) {
            None => Ok(ListChunked::full_null_with_dtype(
                ca.name().clone(),
                ca.len(),
                &DataType::String,
            )),
            Some(val) => {
                let mut builder =
                    ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.len() * 2);

                for pat in patterns.amortized_iter() {
                    match pat {
                        None => builder.append_null(),
                        Some(pat) => {
                            let pat = pat.as_ref();
                            let pat = pat.str()?;
                            let pat = pat.rechunk();
                            let pat = pat.downcast_as_array();
                            let ac = build_ac_arr(pat, ascii_case_insensitive, leftmost)?;
                            push_str(val, &mut builder, &ac, overlapping);
                        },
                    }
                }
                Ok(builder.finish())
            },
        },
        (_, 1) if patterns.has_nulls() => Ok(ListChunked::full_null_with_dtype(
            ca.name().clone(),
            ca.len(),
            &DataType::String,
        )),
        (_, 1) => {
            let patterns = patterns.explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: true,
            })?;
            let patterns = patterns.str()?;
            let ac = build_ac(patterns, ascii_case_insensitive, leftmost)?;
            let mut builder =
                ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.len() * 2);

            for arr in ca.downcast_iter() {
                for opt_val in arr.into_iter() {
                    if let Some(val) = opt_val {
                        push_str(val, &mut builder, &ac, overlapping);
                    } else {
                        builder.append_null();
                    }
                }
            }
            Ok(builder.finish())
        },
        (a, b) if a == b => {
            let mut builder =
                ListStringChunkedBuilder::new(ca.name().clone(), ca.len(), ca.len() * 2);
            let (ca, patterns) = align_chunks_binary(ca, patterns);

            for (arr, pat_arr) in ca.downcast_iter().zip(patterns.downcast_iter()) {
                for z in arr.into_iter().zip(pat_arr) {
                    match z {
                        (None, _) | (_, None) => builder.append_null(),
                        (Some(val), Some(pat)) => {
                            let pat = pat.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                            let ac = build_ac_arr(pat, ascii_case_insensitive, leftmost)?;
                            push_str(val, &mut builder, &ac, overlapping);
                        },
                    }
                }
            }
            Ok(builder.finish())
        },
        (a, b) => polars_bail!(length_mismatch = "str.extract_many", a, b),
    }
}

type B = ListPrimitiveChunkedBuilder<UInt32Type>;
fn push_idx(val: &str, builder: &mut B, ac: &AhoCorasick, overlapping: bool) {
    if overlapping {
        let iter = ac.find_overlapping_iter(val);
        let iter = iter.map(|m| m.start() as u32);
        builder.append_values_iter(iter);
    } else {
        let iter = ac.find_iter(val);
        let iter = iter.map(|m| m.start() as u32);
        builder.append_values_iter(iter);
    }
}

pub fn find_many(
    ca: &StringChunked,
    patterns: &ListChunked,
    ascii_case_insensitive: bool,
    overlapping: bool,
    leftmost: bool,
) -> PolarsResult<ListChunked> {
    polars_ensure!(!overlapping | !leftmost, InvalidOperation: "can not match overlapping patterns when leftmost == True");
    type B = ListPrimitiveChunkedBuilder<UInt32Type>;
    match (ca.len(), patterns.len()) {
        (1, _) => match ca.get(0) {
            None => Ok(ListChunked::full_null_with_dtype(
                ca.name().clone(),
                patterns.len(),
                &DataType::UInt32,
            )),
            Some(val) => {
                let mut builder = B::new(
                    ca.name().clone(),
                    patterns.len(),
                    patterns.len() * 2,
                    DataType::UInt32,
                );
                for pat in patterns.amortized_iter() {
                    match pat {
                        None => builder.append_null(),
                        Some(pat) => {
                            let pat = pat.as_ref();
                            let pat = pat.str()?;
                            let pat = pat.rechunk();
                            let pat = pat.downcast_as_array();
                            let ac = build_ac_arr(pat, ascii_case_insensitive, leftmost)?;
                            push_idx(val, &mut builder, &ac, overlapping);
                        },
                    }
                }
                Ok(builder.finish())
            },
        },
        (_, 1) if patterns.has_nulls() => Ok(ListChunked::full_null_with_dtype(
            ca.name().clone(),
            ca.len(),
            &DataType::UInt32,
        )),
        (_, 1) => {
            let patterns = patterns.explode(ExplodeOptions {
                empty_as_null: false,
                keep_nulls: true,
            })?;
            let patterns = patterns.str()?;
            let ac = build_ac(patterns, ascii_case_insensitive, leftmost)?;
            let mut builder = B::new(ca.name().clone(), ca.len(), ca.len() * 2, DataType::UInt32);

            for opt_val in ca.iter() {
                if let Some(val) = opt_val {
                    push_idx(val, &mut builder, &ac, overlapping);
                } else {
                    builder.append_null();
                }
            }
            Ok(builder.finish())
        },
        (a, b) if a == b => {
            let mut builder = B::new(ca.name().clone(), ca.len(), ca.len() * 2, DataType::UInt32);
            let (ca, patterns) = align_chunks_binary(ca, patterns);

            for (arr, pat_arr) in ca.downcast_iter().zip(patterns.downcast_iter()) {
                for z in arr.into_iter().zip(pat_arr) {
                    match z {
                        (None, _) | (_, None) => builder.append_null(),
                        (Some(val), Some(pat)) => {
                            let pat = pat.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                            let ac = build_ac_arr(pat, ascii_case_insensitive, leftmost)?;
                            push_idx(val, &mut builder, &ac, overlapping);
                        },
                    }
                }
            }
            Ok(builder.finish())
        },
        (a, b) => polars_bail!(length_mismatch = "str.find_many", a, b),
    }
}
