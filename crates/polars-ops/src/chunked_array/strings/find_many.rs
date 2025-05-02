use aho_corasick::{AhoCorasick, AhoCorasickBuilder};
use arrow::array::Utf8ViewArray;
use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

fn build_ac(patterns: &StringChunked, ascii_case_insensitive: bool) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
        .ascii_case_insensitive(ascii_case_insensitive)
        .build(patterns.downcast_iter().flatten().flatten())
        .map_err(|e| polars_err!(ComputeError: "could not build aho corasick automaton {}", e))
}

fn build_ac_arr(
    patterns: &Utf8ViewArray,
    ascii_case_insensitive: bool,
) -> PolarsResult<AhoCorasick> {
    AhoCorasickBuilder::new()
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
    polars_ensure!(
        patterns.len() == 1,
        nyi = "`str.contains_any` with a pattern per row"
    );

    if patterns.has_nulls() {
        return Ok(BooleanChunked::full_null(ca.name().clone(), ca.len()));
    }

    let patterns = patterns.explode(true)?;
    let patterns = patterns.str()?;
    let ac = build_ac(patterns, ascii_case_insensitive)?;

    Ok(unary_elementwise(ca, |opt_val| {
        opt_val.map(|val| ac.find(val).is_some())
    }))
}

pub fn replace_all(
    ca: &StringChunked,
    patterns: &ListChunked,
    replace_with: &ListChunked,
    ascii_case_insensitive: bool,
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

    polars_ensure!(
        patterns.len() == 1 && replace_with.len() == 1,
        nyi = "`str.replace_many` with a pattern per row"
    );

    if patterns.has_nulls() || replace_with.has_nulls() {
        return Ok(StringChunked::full_null(ca.name().clone(), ca.len()));
    }

    let patterns = patterns.explode(true)?;
    let patterns = patterns.str()?;
    let replace_with = replace_with.explode(true)?;
    let replace_with = replace_with.str()?;

    let replace_with = if replace_with.len() == 1 && patterns.len() > 1 {
        replace_with.new_from_index(0, patterns.len())
    } else {
        replace_with.clone()
    };

    polars_ensure!(patterns.len() == replace_with.len(), InvalidOperation: "expected the same amount of patterns as replacement strings");
    polars_ensure!(patterns.null_count() == 0 && replace_with.null_count() == 0, InvalidOperation: "'patterns'/'replace_with' should not have nulls");
    let replace_with = replace_with
        .downcast_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();

    let ac = build_ac(patterns, ascii_case_insensitive)?;

    Ok(unary_elementwise(ca, |opt_val| {
        opt_val.map(|val| ac.replace_all(val, replace_with.as_slice()))
    }))
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
) -> PolarsResult<ListChunked> {
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
                            let ac = build_ac_arr(pat, ascii_case_insensitive)?;
                            push_str(val, &mut builder, &ac, overlapping);
                        },
                    }
                }
                Ok(builder.finish())
            },
        },
        (_, 1) => {
            let patterns = patterns.explode(true)?;
            let patterns = patterns.str()?;
            let ac = build_ac(patterns, ascii_case_insensitive)?;
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
                for z in arr.into_iter().zip(pat_arr.into_iter()) {
                    match z {
                        (None, _) | (_, None) => builder.append_null(),
                        (Some(val), Some(pat)) => {
                            let pat = pat.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                            let ac = build_ac_arr(pat, ascii_case_insensitive)?;
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
) -> PolarsResult<ListChunked> {
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
                            let ac = build_ac_arr(pat, ascii_case_insensitive)?;
                            push_idx(val, &mut builder, &ac, overlapping);
                        },
                    }
                }
                Ok(builder.finish())
            },
        },
        (_, 1) => {
            let patterns = patterns.explode(true)?;
            let patterns = patterns.str()?;
            let ac = build_ac(patterns, ascii_case_insensitive)?;
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
                for z in arr.into_iter().zip(pat_arr.into_iter()) {
                    match z {
                        (None, _) | (_, None) => builder.append_null(),
                        (Some(val), Some(pat)) => {
                            let pat = pat.as_any().downcast_ref::<Utf8ViewArray>().unwrap();
                            let ac = build_ac_arr(pat, ascii_case_insensitive)?;
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
