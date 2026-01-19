use arrow::legacy::error::PolarsResult;
use polars_utils::arena::Node;
use polars_utils::format_pl_smallstr;
use polars_utils::option::OptionTry;

use super::expr_to_ir::ExprToIRContext;
use super::*;
use crate::constants::get_literal_name;
use crate::dsl::{Expr, FunctionExpr};
use crate::plans::conversion::dsl_to_ir::expr_to_ir::to_expr_irs;
use crate::plans::{AExpr, IRFunctionExpr};

pub(super) fn convert_functions(
    input: Vec<Expr>,
    function: FunctionExpr,
    ctx: &mut ExprToIRContext,
) -> PolarsResult<(Node, PlSmallStr)> {
    use {FunctionExpr as F, IRFunctionExpr as I};

    // Converts inputs
    let input_is_empty = input.is_empty();
    let e = to_expr_irs(input, ctx)?;
    let mut set_elementwise = false;

    // Return before converting inputs
    let ir_function = match function {
        #[cfg(feature = "dtype-array")]
        F::ArrayExpr(array_function) => {
            use {ArrayFunction as A, IRArrayFunction as IA};
            I::ArrayExpr(match array_function {
                A::Length => IA::Length,
                A::Min => IA::Min,
                A::Max => IA::Max,
                A::Sum => IA::Sum,
                A::ToList => IA::ToList,
                A::Unique(stable) => IA::Unique(stable),
                A::NUnique => IA::NUnique,
                A::Std(v) => IA::Std(v),
                A::Var(v) => IA::Var(v),
                A::Mean => IA::Mean,
                A::Median => IA::Median,
                #[cfg(feature = "array_any_all")]
                A::Any => IA::Any,
                #[cfg(feature = "array_any_all")]
                A::All => IA::All,
                A::Sort(sort_options) => IA::Sort(sort_options),
                A::Reverse => IA::Reverse,
                A::ArgMin => IA::ArgMin,
                A::ArgMax => IA::ArgMax,
                A::Get(v) => IA::Get(v),
                A::Join(v) => IA::Join(v),
                #[cfg(feature = "is_in")]
                A::Contains { nulls_equal } => IA::Contains { nulls_equal },
                #[cfg(feature = "array_count")]
                A::CountMatches => IA::CountMatches,
                A::Shift => IA::Shift,
                A::Explode(options) => IA::Explode(options),
                A::Concat => IA::Concat,
                A::Slice(offset, length) => IA::Slice(offset, length),
                #[cfg(feature = "array_to_struct")]
                A::ToStruct(ng) => IA::ToStruct(ng),
            })
        },
        F::BinaryExpr(binary_function) => {
            use {BinaryFunction as B, IRBinaryFunction as IB};
            I::BinaryExpr(match binary_function {
                B::Contains => IB::Contains,
                B::StartsWith => IB::StartsWith,
                B::EndsWith => IB::EndsWith,
                #[cfg(feature = "binary_encoding")]
                B::HexDecode(v) => IB::HexDecode(v),
                #[cfg(feature = "binary_encoding")]
                B::HexEncode => IB::HexEncode,
                #[cfg(feature = "binary_encoding")]
                B::Base64Decode(v) => IB::Base64Decode(v),
                #[cfg(feature = "binary_encoding")]
                B::Base64Encode => IB::Base64Encode,
                B::Size => IB::Size,
                #[cfg(feature = "binary_encoding")]
                B::Reinterpret(dtype_expr, v) => {
                    let dtype = dtype_expr.into_datatype(ctx.schema)?;
                    let can_reinterpret_to =
                        |dt: &DataType| dt.is_primitive_numeric() || dt.is_temporal();
                    polars_ensure!(
                        can_reinterpret_to(&dtype) || (
                            dtype.is_array() && dtype.inner_dtype().map(can_reinterpret_to) == Some(true)
                        ),
                        InvalidOperation:
                        "cannot reinterpret binary to dtype {:?}. Only numeric or temporal dtype, or Arrays of these, are supported. Hint: To reinterpret to a nested Array, first reinterpret to a linear Array, and then use reshape",
                        dtype
                    );
                    IB::Reinterpret(dtype, v)
                },
                B::Slice => IB::Slice,
                B::Head => IB::Head,
                B::Tail => IB::Tail,
                B::Get(null_on_oob) => IB::Get(null_on_oob),
            })
        },
        #[cfg(feature = "dtype-categorical")]
        F::Categorical(categorical_function) => {
            use {CategoricalFunction as C, IRCategoricalFunction as IC};
            I::Categorical(match categorical_function {
                C::GetCategories => IC::GetCategories,
                #[cfg(feature = "strings")]
                C::LenBytes => IC::LenBytes,
                #[cfg(feature = "strings")]
                C::LenChars => IC::LenChars,
                #[cfg(feature = "strings")]
                C::StartsWith(v) => IC::StartsWith(v),
                #[cfg(feature = "strings")]
                C::EndsWith(v) => IC::EndsWith(v),
                #[cfg(feature = "strings")]
                C::Slice(s, e) => IC::Slice(s, e),
            })
        },
        #[cfg(feature = "dtype-extension")]
        F::Extension(extension_function) => {
            use {ExtensionFunction as E, IRExtensionFunction as IE};
            I::Extension(match extension_function {
                E::To(dtype) => {
                    let concrete_dtype = dtype.into_datatype(ctx.schema)?;
                    polars_ensure!(matches!(concrete_dtype, DataType::Extension(_, _)),
                        InvalidOperation: "ext.to() requires an Extension dtype, got {concrete_dtype:?}"
                    );
                    IE::To(concrete_dtype)
                },
                E::Storage => IE::Storage,
            })
        },
        F::ListExpr(list_function) => {
            use {IRListFunction as IL, ListFunction as L};
            I::ListExpr(match list_function {
                L::Concat => IL::Concat,
                #[cfg(feature = "is_in")]
                L::Contains { nulls_equal } => IL::Contains { nulls_equal },
                #[cfg(feature = "list_drop_nulls")]
                L::DropNulls => IL::DropNulls,
                #[cfg(feature = "list_sample")]
                L::Sample {
                    is_fraction,
                    with_replacement,
                    shuffle,
                    seed,
                } => IL::Sample {
                    is_fraction,
                    with_replacement,
                    shuffle,
                    seed,
                },
                L::Slice => IL::Slice,
                L::Shift => IL::Shift,
                L::Get(v) => IL::Get(v),
                #[cfg(feature = "list_gather")]
                L::Gather(v) => IL::Gather(v),
                #[cfg(feature = "list_gather")]
                L::GatherEvery => IL::GatherEvery,
                #[cfg(feature = "list_count")]
                L::CountMatches => IL::CountMatches,
                L::Sum => IL::Sum,
                L::Length => IL::Length,
                L::Max => IL::Max,
                L::Min => IL::Min,
                L::Mean => IL::Mean,
                L::Median => IL::Median,
                L::Std(v) => IL::Std(v),
                L::Var(v) => IL::Var(v),
                L::ArgMin => IL::ArgMin,
                L::ArgMax => IL::ArgMax,
                #[cfg(feature = "diff")]
                L::Diff { n, null_behavior } => IL::Diff { n, null_behavior },
                L::Sort(sort_options) => IL::Sort(sort_options),
                L::Reverse => IL::Reverse,
                L::Unique(v) => IL::Unique(v),
                L::NUnique => IL::NUnique,
                #[cfg(feature = "list_sets")]
                L::SetOperation(set_operation) => IL::SetOperation(set_operation),
                #[cfg(feature = "list_any_all")]
                L::Any => IL::Any,
                #[cfg(feature = "list_any_all")]
                L::All => IL::All,
                L::Join(v) => IL::Join(v),
                #[cfg(feature = "dtype-array")]
                L::ToArray(v) => IL::ToArray(v),
                #[cfg(feature = "list_to_struct")]
                L::ToStruct(list_to_struct_args) => IL::ToStruct(list_to_struct_args),
            })
        },
        #[cfg(feature = "strings")]
        F::StringExpr(string_function) => {
            use {IRStringFunction as IS, StringFunction as S};
            I::StringExpr(match string_function {
                S::Format { format, insertions } => {
                    if input_is_empty {
                        polars_ensure!(
                            insertions.is_empty(),
                            ComputeError: "StringFormat didn't get any inputs, format: \"{}\"",
                            format
                        );

                        let out = ctx
                            .arena
                            .add(AExpr::Literal(LiteralValue::Scalar(Scalar::from(format))));

                        return Ok((out, get_literal_name()));
                    } else {
                        IS::Format { format, insertions }
                    }
                },
                #[cfg(feature = "concat_str")]
                S::ConcatHorizontal {
                    delimiter,
                    ignore_nulls,
                } => IS::ConcatHorizontal {
                    delimiter,
                    ignore_nulls,
                },
                #[cfg(feature = "concat_str")]
                S::ConcatVertical {
                    delimiter,
                    ignore_nulls,
                } => IS::ConcatVertical {
                    delimiter,
                    ignore_nulls,
                },
                #[cfg(feature = "regex")]
                S::Contains { literal, strict } => IS::Contains { literal, strict },
                S::CountMatches(v) => IS::CountMatches(v),
                S::EndsWith => IS::EndsWith,
                S::Extract(v) => IS::Extract(v),
                S::ExtractAll => IS::ExtractAll,
                #[cfg(feature = "extract_groups")]
                S::ExtractGroups { dtype, pat } => IS::ExtractGroups { dtype, pat },
                #[cfg(feature = "regex")]
                S::Find { literal, strict } => IS::Find { literal, strict },
                #[cfg(feature = "string_to_integer")]
                S::ToInteger { dtype, strict } => IS::ToInteger { dtype, strict },
                S::LenBytes => IS::LenBytes,
                S::LenChars => IS::LenChars,
                S::Lowercase => IS::Lowercase,
                #[cfg(feature = "extract_jsonpath")]
                S::JsonDecode(dtype) => IS::JsonDecode(dtype.into_datatype(ctx.schema)?),
                #[cfg(feature = "extract_jsonpath")]
                S::JsonPathMatch => IS::JsonPathMatch,
                #[cfg(feature = "regex")]
                S::Replace { n, literal } => IS::Replace { n, literal },
                #[cfg(feature = "string_normalize")]
                S::Normalize { form } => IS::Normalize { form },
                #[cfg(feature = "string_reverse")]
                S::Reverse => IS::Reverse,
                #[cfg(feature = "string_pad")]
                S::PadStart { fill_char } => IS::PadStart { fill_char },
                #[cfg(feature = "string_pad")]
                S::PadEnd { fill_char } => IS::PadEnd { fill_char },
                S::Slice => IS::Slice,
                S::Head => IS::Head,
                S::Tail => IS::Tail,
                #[cfg(feature = "string_encoding")]
                S::HexEncode => IS::HexEncode,
                #[cfg(feature = "binary_encoding")]
                S::HexDecode(v) => IS::HexDecode(v),
                #[cfg(feature = "string_encoding")]
                S::Base64Encode => IS::Base64Encode,
                #[cfg(feature = "binary_encoding")]
                S::Base64Decode(v) => IS::Base64Decode(v),
                S::StartsWith => IS::StartsWith,
                S::StripChars => IS::StripChars,
                S::StripCharsStart => IS::StripCharsStart,
                S::StripCharsEnd => IS::StripCharsEnd,
                S::StripPrefix => IS::StripPrefix,
                S::StripSuffix => IS::StripSuffix,
                #[cfg(feature = "dtype-struct")]
                S::SplitExact { n, inclusive } => IS::SplitExact { n, inclusive },
                #[cfg(feature = "dtype-struct")]
                S::SplitN(v) => IS::SplitN(v),
                #[cfg(feature = "regex")]
                S::SplitRegex { inclusive, strict } => IS::SplitRegex { inclusive, strict },
                #[cfg(feature = "temporal")]
                S::Strptime(data_type, strptime_options) => {
                    let is_column_independent = is_column_independent_aexpr(e[0].node(), ctx.arena);
                    set_elementwise = is_column_independent;
                    let dtype = data_type.into_datatype(ctx.schema)?;
                    polars_ensure!(
                        matches!(dtype,
                            DataType::Date |
                            DataType::Datetime(_, _) |
                            DataType::Time
                        ),
                        InvalidOperation: "`strptime` expects a `date`, `datetime` or `time` got {dtype}"
                    );
                    IS::Strptime(dtype, strptime_options)
                },
                S::Split(v) => IS::Split(v),
                #[cfg(feature = "dtype-decimal")]
                S::ToDecimal { scale } => IS::ToDecimal { scale },
                #[cfg(feature = "nightly")]
                S::Titlecase => IS::Titlecase,
                S::Uppercase => IS::Uppercase,
                #[cfg(feature = "string_pad")]
                S::ZFill => IS::ZFill,
                #[cfg(feature = "find_many")]
                S::ContainsAny {
                    ascii_case_insensitive,
                } => IS::ContainsAny {
                    ascii_case_insensitive,
                },
                #[cfg(feature = "find_many")]
                S::ReplaceMany {
                    ascii_case_insensitive,
                    leftmost,
                } => IS::ReplaceMany {
                    ascii_case_insensitive,
                    leftmost,
                },
                #[cfg(feature = "find_many")]
                S::ExtractMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                } => IS::ExtractMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                },
                #[cfg(feature = "find_many")]
                S::FindMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                } => IS::FindMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                },
                #[cfg(feature = "regex")]
                S::EscapeRegex => IS::EscapeRegex,
            })
        },
        #[cfg(feature = "dtype-struct")]
        F::StructExpr(struct_function) => {
            use {IRStructFunction as IS, StructFunction as S};
            I::StructExpr(match struct_function {
                S::FieldByName(pl_small_str) => IS::FieldByName(pl_small_str),
                S::RenameFields(pl_small_strs) => IS::RenameFields(pl_small_strs),
                S::PrefixFields(pl_small_str) => IS::PrefixFields(pl_small_str),
                S::SuffixFields(pl_small_str) => IS::SuffixFields(pl_small_str),
                S::SelectFields(_) => unreachable!("handled by expression expansion"),
                #[cfg(feature = "json")]
                S::JsonEncode => IS::JsonEncode,
                S::MapFieldNames(f) => IS::MapFieldNames(f),
            })
        },
        #[cfg(feature = "temporal")]
        F::TemporalExpr(temporal_function) => {
            use {IRTemporalFunction as IT, TemporalFunction as T};
            I::TemporalExpr(match temporal_function {
                T::Millennium => IT::Millennium,
                T::Century => IT::Century,
                T::Year => IT::Year,
                T::IsLeapYear => IT::IsLeapYear,
                T::IsoYear => IT::IsoYear,
                T::Quarter => IT::Quarter,
                T::Month => IT::Month,
                T::DaysInMonth => IT::DaysInMonth,
                T::Week => IT::Week,
                T::WeekDay => IT::WeekDay,
                T::Day => IT::Day,
                T::OrdinalDay => IT::OrdinalDay,
                T::Time => IT::Time,
                T::Date => IT::Date,
                T::Datetime => IT::Datetime,
                #[cfg(feature = "dtype-duration")]
                T::Duration(time_unit) => IT::Duration(time_unit),
                T::Hour => IT::Hour,
                T::Minute => IT::Minute,
                T::Second => IT::Second,
                T::Millisecond => IT::Millisecond,
                T::Microsecond => IT::Microsecond,
                T::Nanosecond => IT::Nanosecond,
                #[cfg(feature = "dtype-duration")]
                T::TotalDays { fractional } => IT::TotalDays { fractional },
                #[cfg(feature = "dtype-duration")]
                T::TotalHours { fractional } => IT::TotalHours { fractional },
                #[cfg(feature = "dtype-duration")]
                T::TotalMinutes { fractional } => IT::TotalMinutes { fractional },
                #[cfg(feature = "dtype-duration")]
                T::TotalSeconds { fractional } => IT::TotalSeconds { fractional },
                #[cfg(feature = "dtype-duration")]
                T::TotalMilliseconds { fractional } => IT::TotalMilliseconds { fractional },
                #[cfg(feature = "dtype-duration")]
                T::TotalMicroseconds { fractional } => IT::TotalMicroseconds { fractional },
                #[cfg(feature = "dtype-duration")]
                T::TotalNanoseconds { fractional } => IT::TotalNanoseconds { fractional },
                T::ToString(v) => IT::ToString(v),
                T::CastTimeUnit(time_unit) => IT::CastTimeUnit(time_unit),
                T::WithTimeUnit(time_unit) => IT::WithTimeUnit(time_unit),
                #[cfg(feature = "timezones")]
                T::ConvertTimeZone(time_zone) => IT::ConvertTimeZone(time_zone),
                T::TimeStamp(time_unit) => IT::TimeStamp(time_unit),
                T::Truncate => IT::Truncate,
                #[cfg(feature = "offset_by")]
                T::OffsetBy => IT::OffsetBy,
                #[cfg(feature = "month_start")]
                T::MonthStart => IT::MonthStart,
                #[cfg(feature = "month_end")]
                T::MonthEnd => IT::MonthEnd,
                #[cfg(feature = "timezones")]
                T::BaseUtcOffset => IT::BaseUtcOffset,
                #[cfg(feature = "timezones")]
                T::DSTOffset => IT::DSTOffset,
                T::Round => IT::Round,
                T::Replace => IT::Replace,
                #[cfg(feature = "timezones")]
                T::ReplaceTimeZone(time_zone, non_existent) => {
                    IT::ReplaceTimeZone(time_zone, non_existent)
                },
                T::Combine(time_unit) => IT::Combine(time_unit),
                T::DatetimeFunction {
                    time_unit,
                    time_zone,
                } => IT::DatetimeFunction {
                    time_unit,
                    time_zone,
                },
            })
        },
        #[cfg(feature = "bitwise")]
        F::Bitwise(bitwise_function) => I::Bitwise(match bitwise_function {
            BitwiseFunction::CountOnes => IRBitwiseFunction::CountOnes,
            BitwiseFunction::CountZeros => IRBitwiseFunction::CountZeros,
            BitwiseFunction::LeadingOnes => IRBitwiseFunction::LeadingOnes,
            BitwiseFunction::LeadingZeros => IRBitwiseFunction::LeadingZeros,
            BitwiseFunction::TrailingOnes => IRBitwiseFunction::TrailingOnes,
            BitwiseFunction::TrailingZeros => IRBitwiseFunction::TrailingZeros,
            BitwiseFunction::And => IRBitwiseFunction::And,
            BitwiseFunction::Or => IRBitwiseFunction::Or,
            BitwiseFunction::Xor => IRBitwiseFunction::Xor,
        }),
        F::Boolean(boolean_function) => {
            use {BooleanFunction as B, IRBooleanFunction as IB};
            I::Boolean(match boolean_function {
                B::Any { ignore_nulls } => IB::Any { ignore_nulls },
                B::All { ignore_nulls } => IB::All { ignore_nulls },
                B::IsNull => IB::IsNull,
                B::IsNotNull => IB::IsNotNull,
                B::IsFinite => IB::IsFinite,
                B::IsInfinite => IB::IsInfinite,
                B::IsNan => IB::IsNan,
                B::IsNotNan => IB::IsNotNan,
                #[cfg(feature = "is_first_distinct")]
                B::IsFirstDistinct => IB::IsFirstDistinct,
                #[cfg(feature = "is_last_distinct")]
                B::IsLastDistinct => IB::IsLastDistinct,
                #[cfg(feature = "is_unique")]
                B::IsUnique => IB::IsUnique,
                #[cfg(feature = "is_unique")]
                B::IsDuplicated => IB::IsDuplicated,
                #[cfg(feature = "is_between")]
                B::IsBetween { closed } => IB::IsBetween { closed },
                #[cfg(feature = "is_in")]
                B::IsIn { nulls_equal } => IB::IsIn { nulls_equal },
                #[cfg(feature = "is_close")]
                B::IsClose {
                    abs_tol,
                    rel_tol,
                    nans_equal,
                } => IB::IsClose {
                    abs_tol,
                    rel_tol,
                    nans_equal,
                },
                B::AllHorizontal => {
                    let Some(fst) = e.first() else {
                        return Ok((
                            ctx.arena.add(AExpr::Literal(Scalar::from(true).into())),
                            format_pl_smallstr!("{}", IB::AllHorizontal),
                        ));
                    };

                    if e.len() == 1 {
                        return Ok((
                            AExprBuilder::new_from_node(fst.node())
                                .cast(DataType::Boolean, ctx.arena)
                                .node(),
                            fst.output_name().clone(),
                        ));
                    }

                    // Convert to binary expression as the optimizer understands those.
                    // Don't exceed 128 expressions as we might stackoverflow.
                    if e.len() < 128 {
                        let mut r = AExprBuilder::new_from_node(fst.node());
                        for expr in &e[1..] {
                            r = r.logical_and(expr.node(), ctx.arena);
                        }
                        return Ok((r.node(), fst.output_name().clone()));
                    }

                    IB::AllHorizontal
                },
                B::AnyHorizontal => {
                    // This can be created by col(*).is_null() on empty dataframes.
                    let Some(fst) = e.first() else {
                        return Ok((
                            ctx.arena.add(AExpr::Literal(Scalar::from(false).into())),
                            format_pl_smallstr!("{}", IB::AnyHorizontal),
                        ));
                    };

                    if e.len() == 1 {
                        return Ok((
                            AExprBuilder::new_from_node(fst.node())
                                .cast(DataType::Boolean, ctx.arena)
                                .node(),
                            fst.output_name().clone(),
                        ));
                    }

                    // Convert to binary expression as the optimizer understands those.
                    // Don't exceed 128 expressions as we might stackoverflow.
                    if e.len() < 128 {
                        let mut r = AExprBuilder::new_from_node(fst.node());
                        for expr in &e[1..] {
                            r = r.logical_or(expr.node(), ctx.arena);
                        }
                        return Ok((r.node(), fst.output_name().clone()));
                    }

                    IB::AnyHorizontal
                },
                B::Not => IB::Not,
            })
        },
        #[cfg(feature = "business")]
        F::Business(business_function) => I::Business(match business_function {
            BusinessFunction::BusinessDayCount {
                week_mask,
                holidays,
            } => IRBusinessFunction::BusinessDayCount {
                week_mask,
                holidays,
            },
            BusinessFunction::AddBusinessDay {
                week_mask,
                holidays,
                roll,
            } => IRBusinessFunction::AddBusinessDay {
                week_mask,
                holidays,
                roll,
            },
            BusinessFunction::IsBusinessDay {
                week_mask,
                holidays,
            } => IRBusinessFunction::IsBusinessDay {
                week_mask,
                holidays,
            },
        }),
        #[cfg(feature = "abs")]
        F::Abs => I::Abs,
        F::Negate => I::Negate,
        #[cfg(feature = "hist")]
        F::Hist {
            bin_count,
            include_category,
            include_breakpoint,
        } => I::Hist {
            bin_count,
            include_category,
            include_breakpoint,
        },
        F::NullCount => I::NullCount,
        F::Pow(pow_function) => I::Pow(match pow_function {
            PowFunction::Generic => IRPowFunction::Generic,
            PowFunction::Sqrt => IRPowFunction::Sqrt,
            PowFunction::Cbrt => IRPowFunction::Cbrt,
        }),
        #[cfg(feature = "row_hash")]
        F::Hash(s0, s1, s2, s3) => I::Hash(s0, s1, s2, s3),
        #[cfg(feature = "arg_where")]
        F::ArgWhere => I::ArgWhere,
        #[cfg(feature = "index_of")]
        F::IndexOf => I::IndexOf,
        #[cfg(feature = "search_sorted")]
        F::SearchSorted { side, descending } => I::SearchSorted { side, descending },
        #[cfg(feature = "range")]
        F::Range(range_function) => I::Range(match range_function {
            RangeFunction::IntRange { step, dtype } => {
                let dtype = dtype.into_datatype(ctx.schema)?;
                polars_ensure!(e[0].is_scalar(ctx.arena), ShapeMismatch: "non-scalar start passed to `int_range`");
                polars_ensure!(e[1].is_scalar(ctx.arena), ShapeMismatch: "non-scalar stop passed to `int_range`");
                polars_ensure!(dtype.is_integer(), SchemaMismatch: "non-integer `dtype` passed to `int_range`: '{dtype}'");
                IRRangeFunction::IntRange { step, dtype }
            },
            RangeFunction::IntRanges { dtype } => {
                let dtype = dtype.into_datatype(ctx.schema)?;
                polars_ensure!(dtype.is_integer(), SchemaMismatch: "non-integer `dtype` passed to `int_ranges`: '{dtype}'");
                IRRangeFunction::IntRanges { dtype }
            },
            RangeFunction::LinearSpace { closed } => {
                polars_ensure!(e[0].is_scalar(ctx.arena), ShapeMismatch: "non-scalar start passed to `linear_space`");
                polars_ensure!(e[1].is_scalar(ctx.arena), ShapeMismatch: "non-scalar end passed to `linear_space`");
                polars_ensure!(e[2].is_scalar(ctx.arena), ShapeMismatch: "non-scalar num_samples passed to `linear_space`");
                IRRangeFunction::LinearSpace { closed }
            },
            RangeFunction::LinearSpaces {
                closed,
                array_width,
            } => IRRangeFunction::LinearSpaces {
                closed,
                array_width,
            },
            #[cfg(all(feature = "range", feature = "dtype-date"))]
            RangeFunction::DateRange {
                interval,
                closed,
                arg_type,
            } => {
                use DateRangeArgs::*;
                let arg_names = match arg_type {
                    StartEndSamples => vec!["start", "end", "num_samples"],
                    StartEndInterval => vec!["start", "end"],
                    StartIntervalSamples => vec!["start", "num_samples"],
                    EndIntervalSamples => vec!["end", "num_samples"],
                };
                for (idx, &name) in arg_names.iter().enumerate() {
                    polars_ensure!(e[idx].is_scalar(ctx.arena), ShapeMismatch: "non-scalar {name} passed to `date_range`");
                }
                IRRangeFunction::DateRange {
                    interval,
                    closed,
                    arg_type,
                }
            },
            #[cfg(all(feature = "range", feature = "dtype-date"))]
            RangeFunction::DateRanges {
                interval,
                closed,
                arg_type,
            } => IRRangeFunction::DateRanges {
                interval,
                closed,
                arg_type,
            },
            #[cfg(all(feature = "range", feature = "dtype-datetime"))]
            RangeFunction::DatetimeRange {
                interval,
                closed,
                time_unit,
                time_zone,
                arg_type,
            } => {
                use DateRangeArgs::*;
                let arg_names = match arg_type {
                    StartEndSamples => vec!["start", "end", "num_samples"],
                    StartEndInterval => vec!["start", "end"],
                    StartIntervalSamples => vec!["start", "num_samples"],
                    EndIntervalSamples => vec!["end", "num_samples"],
                };
                for (idx, &name) in arg_names.iter().enumerate() {
                    polars_ensure!(e[idx].is_scalar(ctx.arena), ShapeMismatch: "non-scalar {name} passed to `datetime_range`");
                }
                IRRangeFunction::DatetimeRange {
                    interval,
                    closed,
                    time_unit,
                    time_zone,
                    arg_type,
                }
            },
            #[cfg(all(feature = "range", feature = "dtype-datetime"))]
            RangeFunction::DatetimeRanges {
                interval,
                closed,
                time_unit,
                time_zone,
                arg_type,
            } => IRRangeFunction::DatetimeRanges {
                interval,
                closed,
                time_unit,
                time_zone,
                arg_type,
            },
            #[cfg(all(feature = "range", feature = "dtype-time"))]
            RangeFunction::TimeRange { interval, closed } => {
                polars_ensure!(e[0].is_scalar(ctx.arena), ShapeMismatch: "non-scalar start passed to `time_range`");
                polars_ensure!(e[1].is_scalar(ctx.arena), ShapeMismatch: "non-scalar end passed to `time_range`");
                IRRangeFunction::TimeRange { interval, closed }
            },
            #[cfg(all(feature = "range", feature = "dtype-time"))]
            RangeFunction::TimeRanges { interval, closed } => {
                IRRangeFunction::TimeRanges { interval, closed }
            },
        }),
        #[cfg(feature = "trigonometry")]
        F::Trigonometry(trigonometric_function) => {
            use {IRTrigonometricFunction as IT, TrigonometricFunction as T};
            I::Trigonometry(match trigonometric_function {
                T::Cos => IT::Cos,
                T::Cot => IT::Cot,
                T::Sin => IT::Sin,
                T::Tan => IT::Tan,
                T::ArcCos => IT::ArcCos,
                T::ArcSin => IT::ArcSin,
                T::ArcTan => IT::ArcTan,
                T::Cosh => IT::Cosh,
                T::Sinh => IT::Sinh,
                T::Tanh => IT::Tanh,
                T::ArcCosh => IT::ArcCosh,
                T::ArcSinh => IT::ArcSinh,
                T::ArcTanh => IT::ArcTanh,
                T::Degrees => IT::Degrees,
                T::Radians => IT::Radians,
            })
        },
        #[cfg(feature = "trigonometry")]
        F::Atan2 => I::Atan2,
        #[cfg(feature = "sign")]
        F::Sign => I::Sign,
        F::FillNull => I::FillNull,
        F::FillNullWithStrategy(fill_null_strategy) => I::FillNullWithStrategy(fill_null_strategy),
        #[cfg(feature = "rolling_window")]
        F::RollingExpr { function, options } => {
            use RollingFunction as R;
            use aexpr::IRRollingFunction as IR;

            I::RollingExpr {
                function: match function {
                    R::Min => IR::Min,
                    R::Max => IR::Max,
                    R::Mean => IR::Mean,
                    R::Sum => IR::Sum,
                    R::Quantile => IR::Quantile,
                    R::Var => IR::Var,
                    R::Std => IR::Std,
                    R::Rank => IR::Rank,
                    #[cfg(feature = "moment")]
                    R::Skew => IR::Skew,
                    #[cfg(feature = "moment")]
                    R::Kurtosis => IR::Kurtosis,
                    #[cfg(feature = "cov")]
                    R::CorrCov {
                        corr_cov_options,
                        is_corr,
                    } => IR::CorrCov {
                        corr_cov_options,
                        is_corr,
                    },
                    R::Map(f) => IR::Map(f),
                },
                options,
            }
        },
        #[cfg(feature = "rolling_window_by")]
        F::RollingExprBy {
            function_by,
            options,
        } => {
            use RollingFunctionBy as R;
            use aexpr::IRRollingFunctionBy as IR;

            I::RollingExprBy {
                function_by: match function_by {
                    R::MinBy => IR::MinBy,
                    R::MaxBy => IR::MaxBy,
                    R::MeanBy => IR::MeanBy,
                    R::SumBy => IR::SumBy,
                    R::QuantileBy => IR::QuantileBy,
                    R::VarBy => IR::VarBy,
                    R::StdBy => IR::StdBy,
                    R::RankBy => IR::RankBy,
                },
                options,
            }
        },
        F::Rechunk => I::Rechunk,
        F::Append { upcast } => I::Append { upcast },
        F::ShiftAndFill => {
            polars_ensure!(&e[1].is_scalar(ctx.arena), ShapeMismatch: "'n' must be a scalar value");
            polars_ensure!(&e[2].is_scalar(ctx.arena), ShapeMismatch: "'fill_value' must be a scalar value");
            I::ShiftAndFill
        },
        F::Shift => {
            polars_ensure!(&e[1].is_scalar(ctx.arena), ShapeMismatch: "'n' must be a scalar value");
            I::Shift
        },
        F::DropNans => I::DropNans,
        F::DropNulls => I::DropNulls,
        #[cfg(feature = "mode")]
        F::Mode { maintain_order } => I::Mode { maintain_order },
        #[cfg(feature = "moment")]
        F::Skew(v) => I::Skew(v),
        #[cfg(feature = "moment")]
        F::Kurtosis(l, r) => I::Kurtosis(l, r),
        #[cfg(feature = "dtype-array")]
        F::Reshape(reshape_dimensions) => I::Reshape(reshape_dimensions),
        #[cfg(feature = "repeat_by")]
        F::RepeatBy => I::RepeatBy,
        F::ArgUnique => I::ArgUnique,
        F::ArgMin => I::ArgMin,
        F::ArgMax => I::ArgMax,
        F::ArgSort {
            descending,
            nulls_last,
        } => I::ArgSort {
            descending,
            nulls_last,
        },
        F::MinBy => I::MinBy,
        F::MaxBy => I::MaxBy,
        F::Product => I::Product,
        #[cfg(feature = "rank")]
        F::Rank { options, seed } => I::Rank { options, seed },
        F::Repeat => {
            polars_ensure!(&e[0].is_scalar(ctx.arena), ShapeMismatch: "'value' must be a scalar value");
            polars_ensure!(&e[1].is_scalar(ctx.arena), ShapeMismatch: "'n' must be a scalar value");
            I::Repeat
        },
        #[cfg(feature = "round_series")]
        F::Clip { has_min, has_max } => I::Clip { has_min, has_max },
        #[cfg(feature = "dtype-struct")]
        F::AsStruct => I::AsStruct,
        #[cfg(feature = "top_k")]
        F::TopK { descending } => I::TopK { descending },
        #[cfg(feature = "top_k")]
        F::TopKBy { descending } => I::TopKBy { descending },
        #[cfg(feature = "cum_agg")]
        F::CumCount { reverse } => I::CumCount { reverse },
        #[cfg(feature = "cum_agg")]
        F::CumSum { reverse } => I::CumSum { reverse },
        #[cfg(feature = "cum_agg")]
        F::CumProd { reverse } => I::CumProd { reverse },
        #[cfg(feature = "cum_agg")]
        F::CumMin { reverse } => I::CumMin { reverse },
        #[cfg(feature = "cum_agg")]
        F::CumMax { reverse } => I::CumMax { reverse },
        #[cfg(feature = "cum_agg")]
        F::CumMean { reverse } => I::CumMean { reverse },
        F::Reverse => I::Reverse,
        #[cfg(feature = "dtype-struct")]
        F::ValueCounts {
            sort,
            parallel,
            name,
            normalize,
        } => I::ValueCounts {
            sort,
            parallel,
            name,
            normalize,
        },
        #[cfg(feature = "unique_counts")]
        F::UniqueCounts => I::UniqueCounts,
        #[cfg(feature = "approx_unique")]
        F::ApproxNUnique => I::ApproxNUnique,
        F::Coalesce => I::Coalesce,
        #[cfg(feature = "diff")]
        F::Diff(n) => {
            polars_ensure!(&e[1].is_scalar(ctx.arena), ShapeMismatch: "'n' must be a scalar value");
            I::Diff(n)
        },
        #[cfg(feature = "pct_change")]
        F::PctChange => I::PctChange,
        #[cfg(feature = "interpolate")]
        F::Interpolate(interpolation_method) => I::Interpolate(interpolation_method),
        #[cfg(feature = "interpolate_by")]
        F::InterpolateBy => I::InterpolateBy,
        #[cfg(feature = "log")]
        F::Entropy { base, normalize } => I::Entropy { base, normalize },
        #[cfg(feature = "log")]
        F::Log => I::Log,
        #[cfg(feature = "log")]
        F::Log1p => I::Log1p,
        #[cfg(feature = "log")]
        F::Exp => I::Exp,
        F::Unique(v) => I::Unique(v),
        #[cfg(feature = "round_series")]
        F::Round { decimals, mode } => I::Round { decimals, mode },
        #[cfg(feature = "round_series")]
        F::RoundSF { digits } => I::RoundSF { digits },
        #[cfg(feature = "round_series")]
        F::Floor => I::Floor,
        #[cfg(feature = "round_series")]
        F::Ceil => I::Ceil,
        F::UpperBound => {
            let field = e[0].field(ctx.schema, ctx.arena)?;
            return Ok((
                ctx.arena
                    .add(AExpr::Literal(field.dtype.to_physical().max()?.into())),
                field.name,
            ));
        },
        F::LowerBound => {
            let field = e[0].field(ctx.schema, ctx.arena)?;
            return Ok((
                ctx.arena
                    .add(AExpr::Literal(field.dtype.to_physical().min()?.into())),
                field.name,
            ));
        },
        F::ConcatExpr(v) => I::ConcatExpr(v),
        #[cfg(feature = "cov")]
        F::Correlation { method } => {
            use {CorrelationMethod as C, IRCorrelationMethod as IC};
            I::Correlation {
                method: match method {
                    C::Pearson => IC::Pearson,
                    #[cfg(all(feature = "rank", feature = "propagate_nans"))]
                    C::SpearmanRank(v) => IC::SpearmanRank(v),
                    C::Covariance(v) => IC::Covariance(v),
                },
            }
        },
        #[cfg(feature = "peaks")]
        F::PeakMin => I::PeakMin,
        #[cfg(feature = "peaks")]
        F::PeakMax => I::PeakMax,
        #[cfg(feature = "cutqcut")]
        F::Cut {
            breaks,
            labels,
            left_closed,
            include_breaks,
        } => I::Cut {
            breaks,
            labels,
            left_closed,
            include_breaks,
        },
        #[cfg(feature = "cutqcut")]
        F::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        } => I::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        },
        #[cfg(feature = "rle")]
        F::RLE => I::RLE,
        #[cfg(feature = "rle")]
        F::RLEID => I::RLEID,
        F::ToPhysical => I::ToPhysical,
        #[cfg(feature = "random")]
        F::Random { method, seed } => {
            use {IRRandomMethod as IR, RandomMethod as R};
            I::Random {
                method: match method {
                    R::Shuffle => IR::Shuffle,
                    R::Sample {
                        is_fraction,
                        with_replacement,
                        shuffle,
                    } => IR::Sample {
                        is_fraction,
                        with_replacement,
                        shuffle,
                    },
                },
                seed,
            }
        },
        F::SetSortedFlag(is_sorted) => I::SetSortedFlag(is_sorted),
        #[cfg(feature = "ffi_plugin")]
        F::FfiPlugin {
            flags,
            lib,
            symbol,
            kwargs,
        } => I::FfiPlugin {
            flags,
            lib,
            symbol,
            kwargs,
        },

        F::FoldHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => I::FoldHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.try_map(|dtype| dtype.into_datatype(ctx.schema))?,
        },
        F::ReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => I::ReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.try_map(|dtype| dtype.into_datatype(ctx.schema))?,
        },
        #[cfg(feature = "dtype-struct")]
        F::CumReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => I::CumReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.try_map(|dtype| dtype.into_datatype(ctx.schema))?,
        },
        #[cfg(feature = "dtype-struct")]
        F::CumFoldHorizontal {
            callback,
            returns_scalar,
            return_dtype,
            include_init,
        } => I::CumFoldHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.try_map(|dtype| dtype.into_datatype(ctx.schema))?,
            include_init,
        },

        F::MaxHorizontal => I::MaxHorizontal,
        F::MinHorizontal => I::MinHorizontal,
        F::SumHorizontal { ignore_nulls } => I::SumHorizontal { ignore_nulls },
        F::MeanHorizontal { ignore_nulls } => I::MeanHorizontal { ignore_nulls },
        #[cfg(feature = "ewma")]
        F::EwmMean { options } => I::EwmMean { options },
        #[cfg(feature = "ewma_by")]
        F::EwmMeanBy { half_life } => I::EwmMeanBy { half_life },
        #[cfg(feature = "ewma")]
        F::EwmStd { options } => I::EwmStd { options },
        #[cfg(feature = "ewma")]
        F::EwmVar { options } => I::EwmVar { options },
        #[cfg(feature = "replace")]
        F::Replace => I::Replace,
        #[cfg(feature = "replace")]
        F::ReplaceStrict { return_dtype } => I::ReplaceStrict {
            return_dtype: match return_dtype {
                Some(dtype) => Some(dtype.into_datatype(ctx.schema)?),
                None => None,
            },
        },
        F::GatherEvery { n, offset } => I::GatherEvery { n, offset },
        #[cfg(feature = "reinterpret")]
        F::Reinterpret(v) => I::Reinterpret(v),
        F::ExtendConstant => {
            polars_ensure!(&e[1].is_scalar(ctx.arena), ShapeMismatch: "'value' must be a scalar value");
            polars_ensure!(&e[2].is_scalar(ctx.arena), ShapeMismatch: "'n' must be a scalar value");
            I::ExtendConstant
        },

        F::RowEncode(v) => {
            let dts = e
                .iter()
                .map(|e| Ok(e.dtype(ctx.schema, ctx.arena)?.clone()))
                .collect::<PolarsResult<Vec<_>>>()?;
            I::RowEncode(dts, v)
        },
        #[cfg(feature = "dtype-struct")]
        F::RowDecode(fs, v) => I::RowDecode(
            fs.into_iter()
                .map(|(name, dt_expr)| Ok(Field::new(name, dt_expr.into_datatype(ctx.schema)?)))
                .collect::<PolarsResult<Vec<_>>>()?,
            v,
        ),
    };

    let mut options = ir_function.function_options();
    if set_elementwise {
        options.set_elementwise();
    }

    // Handles special case functions like `struct.field`.
    let output_name = match ir_function.output_name().and_then(|v| v.into_inner()) {
        Some(name) => name,
        None if e.is_empty() => format_pl_smallstr!("{}", &ir_function),
        None => e[0].output_name().clone(),
    };

    let ae_function = AExpr::Function {
        input: e,
        function: ir_function,
        options,
    };
    Ok((ctx.arena.add(ae_function), output_name))
}
