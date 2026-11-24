use aho_corasick::{AhoCorasick, AhoCorasickBuilder, Match, MatchKind};
use polars_arrow::array::{BinaryViewArray, Utf8ViewArray};
use polars_compute::utf8::{byte_to_char_idx, is_utf8_codepoint_start};
use polars_core::prelude::arity::unary_elementwise;
use polars_core::prelude::*;
use polars_core::utils::align_chunks_binary;

fn build_ac<P: AsRef<[u8]>>(
    patterns: impl IntoIterator<Item = P>,
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
        .build(patterns)
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

    let patterns = patterns.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let patterns = patterns.str()?;
    let ac = build_ac(
        patterns.downcast_iter().flatten().flatten(),
        ascii_case_insensitive,
        false,
    )?;

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

    polars_ensure!(
        patterns.len() == 1 && replace_with.len() == 1,
        nyi = "`str.replace_many` with a pattern per row"
    );

    if patterns.has_nulls() || replace_with.has_nulls() {
        return Ok(StringChunked::full_null(ca.name().clone(), ca.len()));
    }

    let patterns = patterns.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let patterns = patterns.str()?;
    let replace_with = replace_with.explode(ExplodeOptions {
        empty_as_null: false,
        keep_nulls: true,
    })?;
    let replace_with = replace_with.str()?;

    let replace_with = replace_with
        .broadcast_to(patterns.len())
        .context("expected the same amount of patterns as replacement strings")?;

    polars_ensure!(patterns.null_count() == 0 && replace_with.null_count() == 0, InvalidOperation: "'patterns'/'replace_with' should not have nulls");
    let replace_with = replace_with
        .downcast_iter()
        .flatten()
        .flatten()
        .collect::<Vec<_>>();

    let ac = build_ac(
        patterns.downcast_iter().flatten().flatten(),
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
                            let ac = build_ac(
                                pat.into_iter().flatten(),
                                ascii_case_insensitive,
                                leftmost,
                            )?;
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
            let ac = build_ac(
                patterns.downcast_iter().flatten().flatten(),
                ascii_case_insensitive,
                leftmost,
            )?;
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
                            let ac = build_ac(
                                pat.into_iter().flatten(),
                                ascii_case_insensitive,
                                leftmost,
                            )?;
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

fn find_ac_impl(
    val: &[u8],
    builder: &mut B,
    ac: &AhoCorasick,
    overlapping: bool,
    char_offsets: bool,
) {
    match (char_offsets, overlapping) {
        (false, false) => builder.append_values_iter(ac.find_iter(val).map(|m| m.start() as u32)),
        (false, true) => {
            builder.append_values_iter(ac.find_overlapping_iter(val).map(|m| m.start() as u32))
        },
        (true, false) => push_char_offsets(val, builder, ac, ac.find_iter(val)),
        (true, true) => push_char_offsets(val, builder, ac, ac.find_overlapping_iter(val)),
    }
}

/// Push the character offsets of the `matches` of `ac` in UTF-8 `val`.
fn push_char_offsets(
    val: &[u8],
    builder: &mut B,
    ac: &AhoCorasick,
    matches: impl Iterator<Item = Match>,
) {
    // Only an empty pattern can match inside a character.
    let may_split_chars = ac.min_pattern_len() == 0;
    // Overlapping matches are ordered by their end, so a start can be before the previous one.
    let mut byte_idx = 0;
    let mut char_idx = 0;
    let iter = matches
        .map(|m| m.start())
        .filter(|&i| !may_split_chars || val.get(i).is_none_or(|&b| is_utf8_codepoint_start(b)))
        .map(|i| {
            if i >= byte_idx {
                char_idx += byte_to_char_idx(&val[byte_idx..], i - byte_idx);
            } else {
                char_idx -= byte_to_char_idx(&val[i..], byte_idx - i);
            }
            byte_idx = i;
            char_idx as u32
        });
    builder.append_values_iter(iter);
}

fn find_many_impl(
    ca: &BinaryChunked,
    patterns: &ListChunked,
    ascii_case_insensitive: bool,
    overlapping: bool,
    leftmost: bool,
    char_offsets: bool,
) -> PolarsResult<ListChunked> {
    polars_ensure!(!overlapping | !leftmost, InvalidOperation: "can not match overlapping patterns when leftmost == True");
    let patterns = patterns.cast(&DataType::List(Box::new(DataType::Binary)))?;
    let patterns = patterns.list()?;
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
                            let pat = pat.as_ref().binary()?.rechunk();
                            let ac = build_ac(
                                pat.downcast_as_array().into_iter().flatten(),
                                ascii_case_insensitive,
                                leftmost,
                            )?;
                            find_ac_impl(val, &mut builder, &ac, overlapping, char_offsets);
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
            let patterns = patterns.binary()?;
            let ac = build_ac(
                patterns.downcast_iter().flatten().flatten(),
                ascii_case_insensitive,
                leftmost,
            )?;
            let mut builder = B::new(ca.name().clone(), ca.len(), ca.len() * 2, DataType::UInt32);

            for opt_val in ca.iter() {
                if let Some(val) = opt_val {
                    find_ac_impl(val, &mut builder, &ac, overlapping, char_offsets);
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
                            let pat = pat.as_any().downcast_ref::<BinaryViewArray>().unwrap();
                            let ac = build_ac(
                                pat.into_iter().flatten(),
                                ascii_case_insensitive,
                                leftmost,
                            )?;
                            find_ac_impl(val, &mut builder, &ac, overlapping, char_offsets);
                        },
                    }
                }
            }
            Ok(builder.finish())
        },
        (a, b) => {
            let name = if char_offsets {
                "str.find_many"
            } else {
                "bin.find_many"
            };
            polars_bail!(length_mismatch = name, a, b)
        },
    }
}

/// Find the character offsets of the matches of `patterns` in each string.
pub fn find_many(
    ca: &StringChunked,
    patterns: &ListChunked,
    ascii_case_insensitive: bool,
    overlapping: bool,
    leftmost: bool,
) -> PolarsResult<ListChunked> {
    find_many_impl(
        &ca.as_binary(),
        patterns,
        ascii_case_insensitive,
        overlapping,
        leftmost,
        true,
    )
}

/// Find the byte offsets of the matches of `patterns` in each binary value.
pub fn bin_find_many(
    ca: &BinaryChunked,
    patterns: &ListChunked,
    ascii_case_insensitive: bool,
    overlapping: bool,
    leftmost: bool,
) -> PolarsResult<ListChunked> {
    find_many_impl(
        ca,
        patterns,
        ascii_case_insensitive,
        overlapping,
        leftmost,
        false,
    )
}
