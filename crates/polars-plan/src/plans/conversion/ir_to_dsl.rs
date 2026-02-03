use super::*;

/// converts a node from the AExpr arena to Expr
#[recursive]
pub fn node_to_expr(node: Node, expr_arena: &Arena<AExpr>) -> Expr {
    let expr = expr_arena.get(node).clone();

    match expr {
        AExpr::Element => Expr::Element,
        AExpr::Explode { expr, options } => Expr::Explode {
            input: Arc::new(node_to_expr(expr, expr_arena)),
            options,
        },
        AExpr::Column(a) => Expr::Column(a),
        #[cfg(feature = "dtype-struct")]
        AExpr::StructField(a) => Expr::Field(Arc::new([a])),
        AExpr::Literal(s) => Expr::Literal(s),
        AExpr::BinaryExpr { left, op, right } => {
            let l = node_to_expr(left, expr_arena);
            let r = node_to_expr(right, expr_arena);
            Expr::BinaryExpr {
                left: Arc::new(l),
                op,
                right: Arc::new(r),
            }
        },
        AExpr::Cast {
            expr,
            dtype,
            options: strict,
        } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Cast {
                expr: Arc::new(exp),
                dtype: dtype.into(),
                options: strict,
            }
        },
        AExpr::Sort { expr, options } => {
            let exp = node_to_expr(expr, expr_arena);
            Expr::Sort {
                expr: Arc::new(exp),
                options,
            }
        },
        AExpr::Gather {
            expr,
            idx,
            returns_scalar,
            null_on_oob,
        } => {
            let expr = node_to_expr(expr, expr_arena);
            let idx = node_to_expr(idx, expr_arena);
            Expr::Gather {
                expr: Arc::new(expr),
                idx: Arc::new(idx),
                returns_scalar,
                null_on_oob,
            }
        },
        AExpr::SortBy {
            expr,
            by,
            sort_options,
        } => {
            let expr = node_to_expr(expr, expr_arena);
            let by = by
                .iter()
                .map(|node| node_to_expr(*node, expr_arena))
                .collect();
            Expr::SortBy {
                expr: Arc::new(expr),
                by,
                sort_options,
            }
        },
        AExpr::Filter { input, by } => {
            let input = node_to_expr(input, expr_arena);
            let by = node_to_expr(by, expr_arena);
            Expr::Filter {
                input: Arc::new(input),
                by: Arc::new(by),
            }
        },
        AExpr::Agg(agg) => match agg {
            IRAggExpr::Min {
                input,
                propagate_nans,
            } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Min {
                    input: Arc::new(exp),
                    propagate_nans,
                }
                .into()
            },
            IRAggExpr::Max {
                input,
                propagate_nans,
            } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Max {
                    input: Arc::new(exp),
                    propagate_nans,
                }
                .into()
            },

            IRAggExpr::Mean(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Mean(Arc::new(exp)).into()
            },
            IRAggExpr::Median(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Median(Arc::new(exp)).into()
            },
            IRAggExpr::NUnique(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::NUnique(Arc::new(exp)).into()
            },
            IRAggExpr::First(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::First(Arc::new(exp)).into()
            },
            IRAggExpr::FirstNonNull(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::FirstNonNull(Arc::new(exp)).into()
            },
            IRAggExpr::Last(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Last(Arc::new(exp)).into()
            },
            IRAggExpr::LastNonNull(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::LastNonNull(Arc::new(exp)).into()
            },
            IRAggExpr::Item { input, allow_empty } => {
                let exp = node_to_expr(input, expr_arena);
                AggExpr::Item {
                    input: Arc::new(exp),
                    allow_empty,
                }
                .into()
            },
            IRAggExpr::Implode(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Implode(Arc::new(exp)).into()
            },
            IRAggExpr::Quantile {
                expr,
                quantile,
                method,
            } => {
                let expr = node_to_expr(expr, expr_arena);
                let quantile = node_to_expr(quantile, expr_arena);
                AggExpr::Quantile {
                    expr: Arc::new(expr),
                    quantile: Arc::new(quantile),
                    method,
                }
                .into()
            },
            IRAggExpr::Sum(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Sum(Arc::new(exp)).into()
            },
            IRAggExpr::Std(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Std(Arc::new(exp), ddof).into()
            },
            IRAggExpr::Var(expr, ddof) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::Var(Arc::new(exp), ddof).into()
            },
            IRAggExpr::AggGroups(expr) => {
                let exp = node_to_expr(expr, expr_arena);
                AggExpr::AggGroups(Arc::new(exp)).into()
            },
            IRAggExpr::Count {
                input,
                include_nulls,
            } => {
                let input = node_to_expr(input, expr_arena);
                AggExpr::Count {
                    input: Arc::new(input),
                    include_nulls,
                }
                .into()
            },
        },
        AExpr::Ternary {
            predicate,
            truthy,
            falsy,
        } => {
            let p = node_to_expr(predicate, expr_arena);
            let t = node_to_expr(truthy, expr_arena);
            let f = node_to_expr(falsy, expr_arena);

            Expr::Ternary {
                predicate: Arc::new(p),
                truthy: Arc::new(t),
                falsy: Arc::new(f),
            }
        },
        AExpr::AnonymousAgg {
            input,
            fmt_str,
            function,
        } => Expr::AnonymousAgg {
            input: expr_irs_to_exprs(input, expr_arena),
            function,
            fmt_str,
        },
        AExpr::AnonymousFunction {
            input,
            function,
            options,
            fmt_str,
        } => Expr::AnonymousFunction {
            input: expr_irs_to_exprs(input, expr_arena),
            function,
            options,
            fmt_str,
        },
        AExpr::Eval {
            expr,
            evaluation,
            variant,
        } => Expr::Eval {
            expr: Arc::new(node_to_expr(expr, expr_arena)),
            evaluation: Arc::new(node_to_expr(evaluation, expr_arena)),
            variant,
        },
        #[cfg(feature = "dtype-struct")]
        AExpr::StructEval { expr, evaluation } => Expr::StructEval {
            expr: Arc::new(node_to_expr(expr, expr_arena)),
            evaluation: expr_irs_to_exprs(evaluation, expr_arena),
        },
        AExpr::Function {
            input,
            function,
            options: _,
        } => {
            let input = expr_irs_to_exprs(input, expr_arena);
            ir_function_to_dsl(input, function)
        },
        #[cfg(feature = "dynamic_group_by")]
        AExpr::Rolling {
            function,
            index_column,
            period,
            offset,
            closed_window,
        } => {
            let function = Arc::new(node_to_expr(function, expr_arena));
            let index_column = Arc::new(node_to_expr(index_column, expr_arena));
            Expr::Rolling {
                function,
                index_column,
                period,
                offset,
                closed_window,
            }
        },
        AExpr::Over {
            function,
            partition_by,
            order_by,
            mapping,
        } => {
            let function = Arc::new(node_to_expr(function, expr_arena));
            let partition_by = nodes_to_exprs(&partition_by, expr_arena);
            let order_by =
                order_by.map(|(n, options)| (Arc::new(node_to_expr(n, expr_arena)), options));
            Expr::Over {
                function,
                partition_by,
                order_by,
                mapping,
            }
        },
        AExpr::Slice {
            input,
            offset,
            length,
        } => Expr::Slice {
            input: Arc::new(node_to_expr(input, expr_arena)),
            offset: Arc::new(node_to_expr(offset, expr_arena)),
            length: Arc::new(node_to_expr(length, expr_arena)),
        },
        AExpr::Len => Expr::Len,
    }
}

fn nodes_to_exprs(nodes: &[Node], expr_arena: &Arena<AExpr>) -> Vec<Expr> {
    nodes.iter().map(|n| node_to_expr(*n, expr_arena)).collect()
}

pub fn ir_function_to_dsl(input: Vec<Expr>, function: IRFunctionExpr) -> Expr {
    use {FunctionExpr as F, IRFunctionExpr as IF};

    let function = match function {
        #[cfg(feature = "dtype-array")]
        IF::ArrayExpr(f) => {
            use {ArrayFunction as A, IRArrayFunction as IA};
            F::ArrayExpr(match f {
                IA::Concat => A::Concat,
                IA::Length => A::Length,
                IA::Min => A::Min,
                IA::Max => A::Max,
                IA::Sum => A::Sum,
                IA::ToList => A::ToList,
                IA::Unique(v) => A::Unique(v),
                IA::NUnique => A::NUnique,
                IA::Std(v) => A::Std(v),
                IA::Var(v) => A::Var(v),
                IA::Mean => A::Mean,
                IA::Median => A::Median,
                #[cfg(feature = "array_any_all")]
                IA::Any => A::Any,
                #[cfg(feature = "array_any_all")]
                IA::All => A::All,
                IA::Sort(v) => A::Sort(v),
                IA::Reverse => A::Reverse,
                IA::ArgMin => A::ArgMin,
                IA::ArgMax => A::ArgMax,
                IA::Get(v) => A::Get(v),
                IA::Join(v) => A::Join(v),
                #[cfg(feature = "is_in")]
                IA::Contains { nulls_equal } => A::Contains { nulls_equal },
                #[cfg(feature = "array_count")]
                IA::CountMatches => A::CountMatches,
                IA::Shift => A::Shift,
                IA::Slice(offset, length) => A::Slice(offset, length),
                IA::Explode(options) => A::Explode(options),
                #[cfg(feature = "array_to_struct")]
                IA::ToStruct(ng) => A::ToStruct(ng),
            })
        },
        IF::BinaryExpr(f) => {
            use {BinaryFunction as B, IRBinaryFunction as IB};
            F::BinaryExpr(match f {
                IB::Contains => B::Contains,
                IB::StartsWith => B::StartsWith,
                IB::EndsWith => B::EndsWith,
                #[cfg(feature = "binary_encoding")]
                IB::HexDecode(v) => B::HexDecode(v),
                #[cfg(feature = "binary_encoding")]
                IB::HexEncode => B::HexEncode,
                #[cfg(feature = "binary_encoding")]
                IB::Base64Decode(v) => B::Base64Decode(v),
                #[cfg(feature = "binary_encoding")]
                IB::Base64Encode => B::Base64Encode,
                IB::Size => B::Size,
                #[cfg(feature = "binary_encoding")]
                IB::Reinterpret(data_type, v) => B::Reinterpret(data_type.into(), v),
                IB::Slice => B::Slice,
                IB::Head => B::Head,
                IB::Tail => B::Tail,
            })
        },
        #[cfg(feature = "dtype-categorical")]
        IF::Categorical(f) => {
            use {CategoricalFunction as C, IRCategoricalFunction as IC};
            F::Categorical(match f {
                IC::GetCategories => C::GetCategories,
                #[cfg(feature = "strings")]
                IC::LenBytes => C::LenBytes,
                #[cfg(feature = "strings")]
                IC::LenChars => C::LenChars,
                #[cfg(feature = "strings")]
                IC::StartsWith(v) => C::StartsWith(v),
                #[cfg(feature = "strings")]
                IC::EndsWith(v) => C::EndsWith(v),
                #[cfg(feature = "strings")]
                IC::Slice(s, l) => C::Slice(s, l),
            })
        },
        #[cfg(feature = "dtype-extension")]
        IF::Extension(f) => {
            use {ExtensionFunction as E, IRExtensionFunction as IE};
            F::Extension(match f {
                IE::To(dtype) => E::To(dtype.into()),
                IE::Storage => E::Storage,
            })
        },
        IF::ListExpr(f) => {
            use {IRListFunction as IL, ListFunction as L};
            F::ListExpr(match f {
                IL::Concat => L::Concat,
                #[cfg(feature = "is_in")]
                IL::Contains { nulls_equal } => L::Contains { nulls_equal },
                #[cfg(feature = "list_drop_nulls")]
                IL::DropNulls => L::DropNulls,
                #[cfg(feature = "list_sample")]
                IL::Sample {
                    is_fraction,
                    with_replacement,
                    shuffle,
                    seed,
                } => L::Sample {
                    is_fraction,
                    with_replacement,
                    shuffle,
                    seed,
                },
                IL::Slice => L::Slice,
                IL::Shift => L::Shift,
                IL::Get(v) => L::Get(v),
                #[cfg(feature = "list_gather")]
                IL::Gather(v) => L::Gather(v),
                #[cfg(feature = "list_gather")]
                IL::GatherEvery => L::GatherEvery,
                #[cfg(feature = "list_count")]
                IL::CountMatches => L::CountMatches,
                IL::Sum => L::Sum,
                IL::Length => L::Length,
                IL::Max => L::Max,
                IL::Min => L::Min,
                IL::Mean => L::Mean,
                IL::Median => L::Median,
                IL::Std(v) => L::Std(v),
                IL::Var(v) => L::Var(v),
                IL::ArgMin => L::ArgMin,
                IL::ArgMax => L::ArgMax,
                #[cfg(feature = "diff")]
                IL::Diff { n, null_behavior } => L::Diff { n, null_behavior },
                IL::Sort(sort_options) => L::Sort(sort_options),
                IL::Reverse => L::Reverse,
                IL::Unique(v) => L::Unique(v),
                IL::NUnique => L::NUnique,
                #[cfg(feature = "list_sets")]
                IL::SetOperation(set_operation) => L::SetOperation(set_operation),
                #[cfg(feature = "list_any_all")]
                IL::Any => L::Any,
                #[cfg(feature = "list_any_all")]
                IL::All => L::All,
                IL::Join(v) => L::Join(v),
                #[cfg(feature = "dtype-array")]
                IL::ToArray(v) => L::ToArray(v),
                #[cfg(feature = "list_to_struct")]
                IL::ToStruct(list_to_struct_args) => L::ToStruct(list_to_struct_args),
            })
        },
        #[cfg(feature = "strings")]
        IF::StringExpr(f) => {
            use {IRStringFunction as IB, StringFunction as B};
            F::StringExpr(match f {
                IB::Format { format, insertions } => B::Format { format, insertions },
                #[cfg(feature = "concat_str")]
                IB::ConcatHorizontal {
                    delimiter,
                    ignore_nulls,
                } => B::ConcatHorizontal {
                    delimiter,
                    ignore_nulls,
                },
                #[cfg(feature = "concat_str")]
                IB::ConcatVertical {
                    delimiter,
                    ignore_nulls,
                } => B::ConcatVertical {
                    delimiter,
                    ignore_nulls,
                },
                #[cfg(feature = "regex")]
                IB::Contains { literal, strict } => B::Contains { literal, strict },
                IB::CountMatches(v) => B::CountMatches(v),
                IB::EndsWith => B::EndsWith,
                IB::Extract(v) => B::Extract(v),
                IB::ExtractAll => B::ExtractAll,
                #[cfg(feature = "extract_groups")]
                IB::ExtractGroups { dtype, pat } => B::ExtractGroups { dtype, pat },
                #[cfg(feature = "regex")]
                IB::Find { literal, strict } => B::Find { literal, strict },
                #[cfg(feature = "string_to_integer")]
                IB::ToInteger { dtype, strict } => B::ToInteger { dtype, strict },
                IB::LenBytes => B::LenBytes,
                IB::LenChars => B::LenChars,
                IB::Lowercase => B::Lowercase,
                #[cfg(feature = "extract_jsonpath")]
                IB::JsonDecode(dtype) => B::JsonDecode(dtype.into()),
                #[cfg(feature = "extract_jsonpath")]
                IB::JsonPathMatch => B::JsonPathMatch,
                #[cfg(feature = "regex")]
                IB::Replace { n, literal } => B::Replace { n, literal },
                #[cfg(feature = "string_normalize")]
                IB::Normalize { form } => B::Normalize { form },
                #[cfg(feature = "string_reverse")]
                IB::Reverse => B::Reverse,
                #[cfg(feature = "string_pad")]
                IB::PadStart { fill_char } => B::PadStart { fill_char },
                #[cfg(feature = "string_pad")]
                IB::PadEnd { fill_char } => B::PadEnd { fill_char },
                IB::Slice => B::Slice,
                IB::Head => B::Head,
                IB::Tail => B::Tail,
                #[cfg(feature = "string_encoding")]
                IB::HexEncode => B::HexEncode,
                #[cfg(feature = "binary_encoding")]
                IB::HexDecode(v) => B::HexDecode(v),
                #[cfg(feature = "string_encoding")]
                IB::Base64Encode => B::Base64Encode,
                #[cfg(feature = "binary_encoding")]
                IB::Base64Decode(v) => B::Base64Decode(v),
                IB::StartsWith => B::StartsWith,
                IB::StripChars => B::StripChars,
                IB::StripCharsStart => B::StripCharsStart,
                IB::StripCharsEnd => B::StripCharsEnd,
                IB::StripPrefix => B::StripPrefix,
                IB::StripSuffix => B::StripSuffix,
                #[cfg(feature = "dtype-struct")]
                IB::SplitExact { n, inclusive } => B::SplitExact { n, inclusive },
                #[cfg(feature = "dtype-struct")]
                IB::SplitN(n) => B::SplitN(n),
                #[cfg(feature = "temporal")]
                IB::Strptime(dtype, strptime_options) => {
                    B::Strptime(dtype.into(), strptime_options)
                },
                IB::Split(v) => B::Split(v),
                #[cfg(feature = "regex")]
                IB::SplitRegex { inclusive, strict } => B::SplitRegex { inclusive, strict },
                #[cfg(feature = "dtype-decimal")]
                IB::ToDecimal { scale } => B::ToDecimal { scale },
                #[cfg(feature = "nightly")]
                IB::Titlecase => B::Titlecase,
                IB::Uppercase => B::Uppercase,
                #[cfg(feature = "string_pad")]
                IB::ZFill => B::ZFill,
                #[cfg(feature = "find_many")]
                IB::ContainsAny {
                    ascii_case_insensitive,
                } => B::ContainsAny {
                    ascii_case_insensitive,
                },
                #[cfg(feature = "find_many")]
                IB::ReplaceMany {
                    ascii_case_insensitive,
                    leftmost,
                } => B::ReplaceMany {
                    ascii_case_insensitive,
                    leftmost,
                },
                #[cfg(feature = "find_many")]
                IB::ExtractMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                } => B::ExtractMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                },
                #[cfg(feature = "find_many")]
                IB::FindMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                } => B::FindMany {
                    ascii_case_insensitive,
                    overlapping,
                    leftmost,
                },
                #[cfg(feature = "regex")]
                IB::EscapeRegex => B::EscapeRegex,
            })
        },
        #[cfg(feature = "dtype-struct")]
        IF::StructExpr(f) => {
            use {IRStructFunction as IB, StructFunction as B};
            F::StructExpr(match f {
                IB::FieldByName(pl_small_str) => B::FieldByName(pl_small_str),
                IB::RenameFields(pl_small_strs) => B::RenameFields(pl_small_strs),
                IB::PrefixFields(pl_small_str) => B::PrefixFields(pl_small_str),
                IB::SuffixFields(pl_small_str) => B::SuffixFields(pl_small_str),
                #[cfg(feature = "json")]
                IB::JsonEncode => B::JsonEncode,
                IB::MapFieldNames(f) => B::MapFieldNames(f),
            })
        },
        #[cfg(feature = "temporal")]
        IF::TemporalExpr(f) => {
            use {IRTemporalFunction as IB, TemporalFunction as B};
            F::TemporalExpr(match f {
                IB::Millennium => B::Millennium,
                IB::Century => B::Century,
                IB::Year => B::Year,
                IB::IsLeapYear => B::IsLeapYear,
                IB::IsoYear => B::IsoYear,
                IB::Quarter => B::Quarter,
                IB::Month => B::Month,
                IB::DaysInMonth => B::DaysInMonth,
                IB::Week => B::Week,
                IB::WeekDay => B::WeekDay,
                IB::Day => B::Day,
                IB::OrdinalDay => B::OrdinalDay,
                IB::Time => B::Time,
                IB::Date => B::Date,
                IB::Datetime => B::Datetime,
                #[cfg(feature = "dtype-duration")]
                IB::Duration(time_unit) => B::Duration(time_unit),
                IB::Hour => B::Hour,
                IB::Minute => B::Minute,
                IB::Second => B::Second,
                IB::Millisecond => B::Millisecond,
                IB::Microsecond => B::Microsecond,
                IB::Nanosecond => B::Nanosecond,
                #[cfg(feature = "dtype-duration")]
                IB::TotalDays { fractional } => B::TotalDays { fractional },
                #[cfg(feature = "dtype-duration")]
                IB::TotalHours { fractional } => B::TotalHours { fractional },
                #[cfg(feature = "dtype-duration")]
                IB::TotalMinutes { fractional } => B::TotalMinutes { fractional },
                #[cfg(feature = "dtype-duration")]
                IB::TotalSeconds { fractional } => B::TotalSeconds { fractional },
                #[cfg(feature = "dtype-duration")]
                IB::TotalMilliseconds { fractional } => B::TotalMilliseconds { fractional },
                #[cfg(feature = "dtype-duration")]
                IB::TotalMicroseconds { fractional } => B::TotalMicroseconds { fractional },
                #[cfg(feature = "dtype-duration")]
                IB::TotalNanoseconds { fractional } => B::TotalNanoseconds { fractional },
                IB::ToString(v) => B::ToString(v),
                IB::CastTimeUnit(time_unit) => B::CastTimeUnit(time_unit),
                IB::WithTimeUnit(time_unit) => B::WithTimeUnit(time_unit),
                #[cfg(feature = "timezones")]
                IB::ConvertTimeZone(time_zone) => B::ConvertTimeZone(time_zone),
                IB::TimeStamp(time_unit) => B::TimeStamp(time_unit),
                IB::Truncate => B::Truncate,
                #[cfg(feature = "offset_by")]
                IB::OffsetBy => B::OffsetBy,
                #[cfg(feature = "month_start")]
                IB::MonthStart => B::MonthStart,
                #[cfg(feature = "month_end")]
                IB::MonthEnd => B::MonthEnd,
                #[cfg(feature = "timezones")]
                IB::BaseUtcOffset => B::BaseUtcOffset,
                #[cfg(feature = "timezones")]
                IB::DSTOffset => B::DSTOffset,
                IB::Round => B::Round,
                IB::Replace => B::Replace,
                #[cfg(feature = "timezones")]
                IB::ReplaceTimeZone(time_zone, non_existent) => {
                    B::ReplaceTimeZone(time_zone, non_existent)
                },
                IB::Combine(time_unit) => B::Combine(time_unit),
                IB::DatetimeFunction {
                    time_unit,
                    time_zone,
                } => B::DatetimeFunction {
                    time_unit,
                    time_zone,
                },
            })
        },
        #[cfg(feature = "bitwise")]
        IF::Bitwise(f) => {
            use {BitwiseFunction as B, IRBitwiseFunction as IB};
            F::Bitwise(match f {
                IB::CountOnes => B::CountOnes,
                IB::CountZeros => B::CountZeros,
                IB::LeadingOnes => B::LeadingOnes,
                IB::LeadingZeros => B::LeadingZeros,
                IB::TrailingOnes => B::TrailingOnes,
                IB::TrailingZeros => B::TrailingZeros,
                IB::And => B::And,
                IB::Or => B::Or,
                IB::Xor => B::Xor,
            })
        },
        IF::Boolean(f) => {
            use {BooleanFunction as B, IRBooleanFunction as IB};
            F::Boolean(match f {
                IB::Any { ignore_nulls } => B::Any { ignore_nulls },
                IB::All { ignore_nulls } => B::All { ignore_nulls },
                IB::IsNull => B::IsNull,
                IB::IsNotNull => B::IsNotNull,
                IB::IsFinite => B::IsFinite,
                IB::IsInfinite => B::IsInfinite,
                IB::IsNan => B::IsNan,
                IB::IsNotNan => B::IsNotNan,
                #[cfg(feature = "is_first_distinct")]
                IB::IsFirstDistinct => B::IsFirstDistinct,
                #[cfg(feature = "is_last_distinct")]
                IB::IsLastDistinct => B::IsLastDistinct,
                #[cfg(feature = "is_unique")]
                IB::IsUnique => B::IsUnique,
                #[cfg(feature = "is_unique")]
                IB::IsDuplicated => B::IsDuplicated,
                #[cfg(feature = "is_between")]
                IB::IsBetween { closed } => B::IsBetween { closed },
                #[cfg(feature = "is_in")]
                IB::IsIn { nulls_equal } => B::IsIn { nulls_equal },
                #[cfg(feature = "is_close")]
                IB::IsClose {
                    abs_tol,
                    rel_tol,
                    nans_equal,
                } => B::IsClose {
                    abs_tol,
                    rel_tol,
                    nans_equal,
                },
                IB::AllHorizontal => B::AllHorizontal,
                IB::AnyHorizontal => B::AnyHorizontal,
                IB::Not => B::Not,
            })
        },
        #[cfg(feature = "business")]
        IF::Business(f) => {
            use {BusinessFunction as B, IRBusinessFunction as IB};
            F::Business(match f {
                IB::BusinessDayCount {
                    week_mask,
                    holidays,
                } => B::BusinessDayCount {
                    week_mask,
                    holidays,
                },
                IB::AddBusinessDay {
                    week_mask,
                    holidays,
                    roll,
                } => B::AddBusinessDay {
                    week_mask,
                    holidays,
                    roll,
                },
                IB::IsBusinessDay {
                    week_mask,
                    holidays,
                } => B::IsBusinessDay {
                    week_mask,
                    holidays,
                },
            })
        },
        #[cfg(feature = "abs")]
        IF::Abs => F::Abs,
        IF::Negate => F::Negate,
        #[cfg(feature = "hist")]
        IF::Hist {
            bin_count,
            include_category,
            include_breakpoint,
        } => F::Hist {
            bin_count,
            include_category,
            include_breakpoint,
        },
        IF::NullCount => F::NullCount,
        IF::Pow(f) => {
            use {IRPowFunction as IP, PowFunction as P};
            F::Pow(match f {
                IP::Generic => P::Generic,
                IP::Sqrt => P::Sqrt,
                IP::Cbrt => P::Cbrt,
            })
        },
        #[cfg(feature = "row_hash")]
        IF::Hash(s0, s1, s2, s3) => F::Hash(s0, s1, s2, s3),
        #[cfg(feature = "arg_where")]
        IF::ArgWhere => F::ArgWhere,
        #[cfg(feature = "index_of")]
        IF::IndexOf => F::IndexOf,
        #[cfg(feature = "search_sorted")]
        IF::SearchSorted { side, descending } => F::SearchSorted { side, descending },
        #[cfg(feature = "range")]
        IF::Range(f) => {
            use {IRRangeFunction as IR, RangeFunction as R};
            F::Range(match f {
                IR::IntRange { step, dtype } => R::IntRange {
                    step,
                    dtype: dtype.into(),
                },
                IR::IntRanges { dtype } => R::IntRanges {
                    dtype: dtype.into(),
                },
                IR::LinearSpace { closed } => R::LinearSpace { closed },
                IR::LinearSpaces {
                    closed,
                    array_width,
                } => R::LinearSpaces {
                    closed,
                    array_width,
                },
                #[cfg(all(feature = "range", feature = "dtype-date"))]
                IR::DateRange {
                    interval,
                    closed,
                    arg_type,
                } => R::DateRange {
                    interval,
                    closed,
                    arg_type,
                },
                #[cfg(all(feature = "range", feature = "dtype-date"))]
                IR::DateRanges {
                    interval,
                    closed,
                    arg_type,
                } => R::DateRanges {
                    interval,
                    closed,
                    arg_type,
                },
                #[cfg(all(feature = "range", feature = "dtype-datetime"))]
                IR::DatetimeRange {
                    interval,
                    closed,
                    time_unit,
                    time_zone,
                    arg_type,
                } => R::DatetimeRange {
                    interval,
                    closed,
                    time_unit,
                    time_zone,
                    arg_type,
                },
                #[cfg(all(feature = "range", feature = "dtype-datetime"))]
                IR::DatetimeRanges {
                    interval,
                    closed,
                    time_unit,
                    time_zone,
                    arg_type,
                } => R::DatetimeRanges {
                    interval,
                    closed,
                    time_unit,
                    time_zone,
                    arg_type,
                },
                #[cfg(feature = "dtype-time")]
                IR::TimeRange { interval, closed } => R::TimeRange { interval, closed },
                #[cfg(feature = "dtype-time")]
                IR::TimeRanges { interval, closed } => R::TimeRanges { interval, closed },
            })
        },
        #[cfg(feature = "trigonometry")]
        IF::Trigonometry(f) => {
            use {IRTrigonometricFunction as IT, TrigonometricFunction as T};
            F::Trigonometry(match f {
                IT::Cos => T::Cos,
                IT::Cot => T::Cot,
                IT::Sin => T::Sin,
                IT::Tan => T::Tan,
                IT::ArcCos => T::ArcCos,
                IT::ArcSin => T::ArcSin,
                IT::ArcTan => T::ArcTan,
                IT::Cosh => T::Cosh,
                IT::Sinh => T::Sinh,
                IT::Tanh => T::Tanh,
                IT::ArcCosh => T::ArcCosh,
                IT::ArcSinh => T::ArcSinh,
                IT::ArcTanh => T::ArcTanh,
                IT::Degrees => T::Degrees,
                IT::Radians => T::Radians,
            })
        },
        #[cfg(feature = "trigonometry")]
        IF::Atan2 => F::Atan2,
        #[cfg(feature = "sign")]
        IF::Sign => F::Sign,
        IF::FillNull => F::FillNull,
        IF::FillNullWithStrategy(strategy) => F::FillNullWithStrategy(strategy),
        #[cfg(feature = "rolling_window")]
        IF::RollingExpr { function, options } => {
            use {IRRollingFunction as IR, RollingFunction as R};
            FunctionExpr::RollingExpr {
                function: match function {
                    IR::Min => R::Min,
                    IR::Max => R::Max,
                    IR::Mean => R::Mean,
                    IR::Sum => R::Sum,
                    IR::Quantile => R::Quantile,
                    IR::Var => R::Var,
                    IR::Std => R::Std,
                    IR::Rank => R::Rank,
                    #[cfg(feature = "moment")]
                    IR::Skew => R::Skew,
                    #[cfg(feature = "moment")]
                    IR::Kurtosis => R::Kurtosis,
                    #[cfg(feature = "cov")]
                    IR::CorrCov {
                        corr_cov_options,
                        is_corr,
                    } => R::CorrCov {
                        corr_cov_options,
                        is_corr,
                    },
                    IR::Map(f) => R::Map(f),
                },
                options,
            }
        },
        #[cfg(feature = "rolling_window_by")]
        IF::RollingExprBy {
            function_by,
            options,
        } => {
            use {IRRollingFunctionBy as IR, RollingFunctionBy as R};
            FunctionExpr::RollingExprBy {
                function_by: match function_by {
                    IR::MinBy => R::MinBy,
                    IR::MaxBy => R::MaxBy,
                    IR::MeanBy => R::MeanBy,
                    IR::SumBy => R::SumBy,
                    IR::QuantileBy => R::QuantileBy,
                    IR::VarBy => R::VarBy,
                    IR::StdBy => R::StdBy,
                    IR::RankBy => R::RankBy,
                },
                options,
            }
        },
        IF::Rechunk => F::Rechunk,
        IF::Append { upcast } => F::Append { upcast },
        IF::ShiftAndFill => F::ShiftAndFill,
        IF::Shift => F::Shift,
        IF::DropNans => F::DropNans,
        IF::DropNulls => F::DropNulls,
        #[cfg(feature = "mode")]
        IF::Mode { maintain_order } => F::Mode { maintain_order },
        #[cfg(feature = "moment")]
        IF::Skew(v) => F::Skew(v),
        #[cfg(feature = "moment")]
        IF::Kurtosis(fisher, bias) => F::Kurtosis(fisher, bias),
        #[cfg(feature = "dtype-array")]
        IF::Reshape(dims) => F::Reshape(dims),
        #[cfg(feature = "repeat_by")]
        IF::RepeatBy => F::RepeatBy,
        IF::ArgUnique => F::ArgUnique,
        IF::ArgMin => F::ArgMin,
        IF::ArgMax => F::ArgMax,
        IF::ArgSort {
            descending,
            nulls_last,
        } => F::ArgSort {
            descending,
            nulls_last,
        },
        IF::MinBy => F::MinBy,
        IF::MaxBy => F::MaxBy,
        IF::Product => F::Product,
        #[cfg(feature = "rank")]
        IF::Rank { options, seed } => F::Rank { options, seed },
        IF::Repeat => F::Repeat,
        #[cfg(feature = "round_series")]
        IF::Clip { has_min, has_max } => F::Clip { has_min, has_max },
        #[cfg(feature = "dtype-struct")]
        IF::AsStruct => F::AsStruct,
        #[cfg(feature = "top_k")]
        IF::TopK { descending } => F::TopK { descending },
        #[cfg(feature = "top_k")]
        IF::TopKBy { descending } => F::TopKBy { descending },
        #[cfg(feature = "cum_agg")]
        IF::CumCount { reverse } => F::CumCount { reverse },
        #[cfg(feature = "cum_agg")]
        IF::CumSum { reverse } => F::CumSum { reverse },
        #[cfg(feature = "cum_agg")]
        IF::CumProd { reverse } => F::CumProd { reverse },
        #[cfg(feature = "cum_agg")]
        IF::CumMin { reverse } => F::CumMin { reverse },
        #[cfg(feature = "cum_agg")]
        IF::CumMax { reverse } => F::CumMax { reverse },
        IF::Reverse => F::Reverse,
        #[cfg(feature = "dtype-struct")]
        IF::ValueCounts {
            sort,
            parallel,
            name,
            normalize,
        } => F::ValueCounts {
            sort,
            parallel,
            name,
            normalize,
        },
        #[cfg(feature = "unique_counts")]
        IF::UniqueCounts => F::UniqueCounts,
        #[cfg(feature = "approx_unique")]
        IF::ApproxNUnique => F::ApproxNUnique,
        IF::Coalesce => F::Coalesce,
        #[cfg(feature = "diff")]
        IF::Diff(nb) => F::Diff(nb),
        #[cfg(feature = "pct_change")]
        IF::PctChange => F::PctChange,
        #[cfg(feature = "interpolate")]
        IF::Interpolate(m) => F::Interpolate(m),
        #[cfg(feature = "interpolate_by")]
        IF::InterpolateBy => F::InterpolateBy,
        #[cfg(feature = "log")]
        IF::Entropy { base, normalize } => F::Entropy { base, normalize },
        #[cfg(feature = "log")]
        IF::Log => F::Log,
        #[cfg(feature = "log")]
        IF::Log1p => F::Log1p,
        #[cfg(feature = "log")]
        IF::Exp => F::Exp,
        IF::Unique(v) => F::Unique(v),
        #[cfg(feature = "round_series")]
        IF::Round { decimals, mode } => F::Round { decimals, mode },
        #[cfg(feature = "round_series")]
        IF::RoundSF { digits } => F::RoundSF { digits },
        #[cfg(feature = "round_series")]
        IF::Floor => F::Floor,
        #[cfg(feature = "round_series")]
        IF::Ceil => F::Ceil,
        #[cfg(feature = "fused")]
        IF::Fused(f) => {
            assert_eq!(input.len(), 3);
            let mut input = input.into_iter();
            let fst = input.next().unwrap();
            let snd = input.next().unwrap();
            let trd = input.next().unwrap();
            return match f {
                FusedOperator::MultiplyAdd => (fst * snd) + trd,
                FusedOperator::SubMultiply => fst - (snd * trd),
                FusedOperator::MultiplySub => (fst * snd) - trd,
            };
        },
        IF::ConcatExpr(v) => F::ConcatExpr(v),
        #[cfg(feature = "cov")]
        IF::Correlation { method } => {
            use {CorrelationMethod as C, IRCorrelationMethod as IC};
            F::Correlation {
                method: match method {
                    IC::Pearson => C::Pearson,
                    #[cfg(all(feature = "rank", feature = "propagate_nans"))]
                    IC::SpearmanRank(v) => C::SpearmanRank(v),
                    IC::Covariance(v) => C::Covariance(v),
                },
            }
        },
        #[cfg(feature = "peaks")]
        IF::PeakMin => F::PeakMin,
        #[cfg(feature = "peaks")]
        IF::PeakMax => F::PeakMax,
        #[cfg(feature = "cutqcut")]
        IF::Cut {
            breaks,
            labels,
            left_closed,
            include_breaks,
        } => F::Cut {
            breaks,
            labels,
            left_closed,
            include_breaks,
        },
        #[cfg(feature = "cutqcut")]
        IF::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        } => F::QCut {
            probs,
            labels,
            left_closed,
            allow_duplicates,
            include_breaks,
        },
        #[cfg(feature = "rle")]
        IF::RLE => F::RLE,
        #[cfg(feature = "rle")]
        IF::RLEID => F::RLEID,
        IF::ToPhysical => F::ToPhysical,
        #[cfg(feature = "random")]
        IF::Random { method, seed } => {
            use {IRRandomMethod as IR, RandomMethod as R};
            F::Random {
                method: match method {
                    IR::Shuffle => R::Shuffle,
                    IR::Sample {
                        is_fraction,
                        with_replacement,
                        shuffle,
                    } => R::Sample {
                        is_fraction,
                        with_replacement,
                        shuffle,
                    },
                },
                seed,
            }
        },
        IF::SetSortedFlag(s) => F::SetSortedFlag(s),
        #[cfg(feature = "ffi_plugin")]
        IF::FfiPlugin {
            flags,
            lib,
            symbol,
            kwargs,
        } => F::FfiPlugin {
            flags,
            lib,
            symbol,
            kwargs,
        },

        IF::FoldHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => F::FoldHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.map(DataTypeExpr::Literal),
        },
        IF::ReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => F::ReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.map(DataTypeExpr::Literal),
        },
        #[cfg(feature = "dtype-struct")]
        IF::CumReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype,
        } => F::CumReduceHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.map(DataTypeExpr::Literal),
        },
        #[cfg(feature = "dtype-struct")]
        IF::CumFoldHorizontal {
            callback,
            returns_scalar,
            return_dtype,
            include_init,
        } => F::CumFoldHorizontal {
            callback,
            returns_scalar,
            return_dtype: return_dtype.map(DataTypeExpr::Literal),
            include_init,
        },

        IF::MaxHorizontal => F::MaxHorizontal,
        IF::MinHorizontal => F::MinHorizontal,
        IF::SumHorizontal { ignore_nulls } => F::SumHorizontal { ignore_nulls },
        IF::MeanHorizontal { ignore_nulls } => F::MeanHorizontal { ignore_nulls },
        #[cfg(feature = "ewma")]
        IF::EwmMean { options } => F::EwmMean { options },
        #[cfg(feature = "ewma_by")]
        IF::EwmMeanBy { half_life } => F::EwmMeanBy { half_life },
        #[cfg(feature = "ewma")]
        IF::EwmStd { options } => F::EwmStd { options },
        #[cfg(feature = "ewma")]
        IF::EwmVar { options } => F::EwmVar { options },
        #[cfg(feature = "replace")]
        IF::Replace => F::Replace,
        #[cfg(feature = "replace")]
        IF::ReplaceStrict { return_dtype } => F::ReplaceStrict {
            return_dtype: return_dtype.map(Into::into),
        },
        IF::GatherEvery { n, offset } => F::GatherEvery { n, offset },
        #[cfg(feature = "reinterpret")]
        IF::Reinterpret(v) => F::Reinterpret(v),
        IF::ExtendConstant => F::ExtendConstant,

        IF::RowEncode(_, v) => F::RowEncode(v),
        #[cfg(feature = "dtype-struct")]
        IF::RowDecode(fs, v) => F::RowDecode(
            fs.into_iter().map(|f| (f.name, f.dtype.into())).collect(),
            v,
        ),
    };

    Expr::Function { input, function }
}
