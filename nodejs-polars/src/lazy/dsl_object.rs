use crate::lazy::dsl;
use crate::prelude::JsResult;
use napi::JsObject;

impl dsl::JsWhen {
    pub fn to_object(env: &napi::Env) -> JsResult<JsObject> {
        let mut when = env.create_object()?;
        when.create_named_method("then", dsl::when_then)?;
        Ok(when)
    }
}

impl dsl::JsWhenThen {
    pub fn to_object(env: &napi::Env) -> JsResult<JsObject> {
        let mut when_then = env.create_object()?;
        when_then.define_properties(&[
            napi::Property::new(env, "when")?.with_method(dsl::when_then_when),
            napi::Property::new(env, "otherwise")?.with_method(dsl::when_then_otherwise),
        ])?;
        Ok(when_then)
    }
}

impl dsl::JsWhenThenThen {
    pub fn to_object(env: &napi::Env) -> JsResult<JsObject> {
        let mut when_then_then = env.create_object()?;
        when_then_then.define_properties(&[
            napi::Property::new(env, "then")?.with_method(dsl::when_then_then_then),
            napi::Property::new(env, "when")?.with_method(dsl::when_then_then_when),
            napi::Property::new(env, "otherwise")?.with_method(dsl::when_then_then_otherwise),
        ])?;
        Ok(when_then_then)
    }
}

impl dsl::JsExpr {
    pub fn to_object(env: &napi::Env) -> JsResult<JsObject> {
        let mut expr = env.create_object()?;
        let mut lst_obj = env.create_object()?;
        let mut date_obj = env.create_object()?;
        let mut str_obj = env.create_object()?;

        lst_obj.define_properties(&[
            napi::Property::new(env, "get")?.with_method(dsl::lst_get),
            napi::Property::new(env, "lengths")?.with_method(dsl::lst_lengths),
            napi::Property::new(env, "join")?.with_method(dsl::lst_join),
            napi::Property::new(env, "max")?.with_method(dsl::lst_max),
            napi::Property::new(env, "mean")?.with_method(dsl::lst_mean),
            napi::Property::new(env, "min")?.with_method(dsl::lst_min),
            napi::Property::new(env, "reverse")?.with_method(dsl::lst_reverse),
            napi::Property::new(env, "sort")?.with_method(dsl::lst_sort),
            napi::Property::new(env, "sum")?.with_method(dsl::lst_sum),
            napi::Property::new(env, "unique")?.with_method(dsl::lst_unique),
        ])?;
        str_obj.define_properties(&[
            napi::Property::new(env, "concat")?.with_method(dsl::str_concat),
            napi::Property::new(env, "contains")?.with_method(dsl::str_contains),
            napi::Property::new(env, "extract")?.with_method(dsl::str_extract),
            napi::Property::new(env, "jsonPathMatch")?.with_method(dsl::str_json_path_match),
            napi::Property::new(env, "lengths")?.with_method(dsl::str_lengths),
            napi::Property::new(env, "parseDate")?.with_method(dsl::str_parse_date),
            napi::Property::new(env, "parseDateTime")?.with_method(dsl::str_parse_datetime),
            napi::Property::new(env, "replace")?.with_method(dsl::str_replace),
            napi::Property::new(env, "replaceAll")?.with_method(dsl::str_replace_all),
            napi::Property::new(env, "slice")?.with_method(dsl::str_slice),
            napi::Property::new(env, "split")?.with_method(dsl::str_split),
            napi::Property::new(env, "toLowerCase")?.with_method(dsl::str_to_lowercase),
            napi::Property::new(env, "toUpperCase")?.with_method(dsl::str_to_uppercase),
            napi::Property::new(env, "encodeHex")?.with_method(dsl::hex_encode),
            napi::Property::new(env, "decodeHex")?.with_method(dsl::hex_decode),
            napi::Property::new(env, "encodeBase64")?.with_method(dsl::base64_encode),
            napi::Property::new(env, "decodeBase64")?.with_method(dsl::base64_decode),
        ])?;
        date_obj.define_properties(&[
            napi::Property::new(env, "day")?.with_method(dsl::day),
            napi::Property::new(env, "hour")?.with_method(dsl::hour),
            napi::Property::new(env, "minute")?.with_method(dsl::minute),
            napi::Property::new(env, "month")?.with_method(dsl::month),
            napi::Property::new(env, "nanosecond")?.with_method(dsl::nanosecond),
            napi::Property::new(env, "ordinalDay")?.with_method(dsl::ordinal_day),
            napi::Property::new(env, "second")?.with_method(dsl::second),
            napi::Property::new(env, "strftime")?.with_method(dsl::strftime),
            napi::Property::new(env, "timestamp")?.with_method(dsl::timestamp),
            napi::Property::new(env, "week")?.with_method(dsl::week),
            napi::Property::new(env, "weekday")?.with_method(dsl::weekday),
            napi::Property::new(env, "year")?.with_method(dsl::year),
        ])?;

        expr.define_properties(&[
            napi::Property::new(env, "str")?.with_value(str_obj),
            napi::Property::new(env, "lst")?.with_value(lst_obj),
            napi::Property::new(env, "date")?.with_value(date_obj),
            napi::Property::new(env, "add")?.with_method(dsl::add),
            napi::Property::new(env, "sub")?.with_method(dsl::sub),
            napi::Property::new(env, "div")?.with_method(dsl::div),
            napi::Property::new(env, "mul")?.with_method(dsl::mul),
            napi::Property::new(env, "rem")?.with_method(dsl::rem),
            napi::Property::new(env, "as_str")?.with_method(dsl::as_str),
            napi::Property::new(env, "abs")?.with_method(dsl::abs),
            napi::Property::new(env, "aggGroups")?.with_method(dsl::agg_groups),
            napi::Property::new(env, "alias")?.with_method(dsl::alias),
            napi::Property::new(env, "and")?.with_method(dsl::and),
            napi::Property::new(env, "argMax")?.with_method(dsl::arg_max),
            napi::Property::new(env, "argSort")?.with_method(dsl::arg_sort),
            napi::Property::new(env, "argUnique")?.with_method(dsl::arg_unique),
            napi::Property::new(env, "argMin")?.with_method(dsl::arg_min),
            napi::Property::new(env, "backwardFill")?.with_method(dsl::backward_fill),
            napi::Property::new(env, "cast")?.with_method(dsl::cast),
            napi::Property::new(env, "ceil")?.with_method(dsl::ceil),
            napi::Property::new(env, "clip")?.with_method(dsl::clip),
            napi::Property::new(env, "count")?.with_method(dsl::count),
            napi::Property::new(env, "cumCount")?.with_method(dsl::cumcount),
            napi::Property::new(env, "cumMax")?.with_method(dsl::cummax),
            napi::Property::new(env, "cumMin")?.with_method(dsl::cummin),
            napi::Property::new(env, "cumProd")?.with_method(dsl::cumprod),
            napi::Property::new(env, "cumSum")?.with_method(dsl::cumsum),
            napi::Property::new(env, "diff")?.with_method(dsl::diff),
            napi::Property::new(env, "dot")?.with_method(dsl::dot),
            napi::Property::new(env, "eq")?.with_method(dsl::eq),
            napi::Property::new(env, "exclude")?.with_method(dsl::exclude),
            napi::Property::new(env, "explode")?.with_method(dsl::explode),
            napi::Property::new(env, "extendConstant")?.with_method(dsl::extend_constant),
            napi::Property::new(env, "fillNan")?.with_method(dsl::fill_nan),
            napi::Property::new(env, "fillNullWithStrategy")?
                .with_method(dsl::fill_null_with_strategy),
            napi::Property::new(env, "fillNull")?.with_method(dsl::fill_null),
            napi::Property::new(env, "filter")?.with_method(dsl::filter),
            napi::Property::new(env, "first")?.with_method(dsl::first),
            napi::Property::new(env, "floor")?.with_method(dsl::floor),
            napi::Property::new(env, "forwardFill")?.with_method(dsl::forward_fill),
            napi::Property::new(env, "gtEq")?.with_method(dsl::gt_eq),
            napi::Property::new(env, "gt")?.with_method(dsl::gt),
            napi::Property::new(env, "hash")?.with_method(dsl::hash),
            napi::Property::new(env, "head")?.with_method(dsl::head),
            napi::Property::new(env, "interpolate")?.with_method(dsl::interpolate),
            napi::Property::new(env, "isDuplicated")?.with_method(dsl::is_duplicated),
            napi::Property::new(env, "isFinite")?.with_method(dsl::is_finite),
            napi::Property::new(env, "isFirst")?.with_method(dsl::is_first),
            napi::Property::new(env, "isIn")?.with_method(dsl::is_in),
            napi::Property::new(env, "isInfinite")?.with_method(dsl::is_infinite),
            napi::Property::new(env, "isNan")?.with_method(dsl::is_nan),
            napi::Property::new(env, "isNotNan")?.with_method(dsl::is_not_nan),
            napi::Property::new(env, "isNotNull")?.with_method(dsl::is_not_null),
            napi::Property::new(env, "isNull")?.with_method(dsl::is_null),
            napi::Property::new(env, "isUnique")?.with_method(dsl::is_unique),
            napi::Property::new(env, "keepName")?.with_method(dsl::keep_name),
            napi::Property::new(env, "kurtosis")?.with_method(dsl::kurtosis),
            napi::Property::new(env, "last")?.with_method(dsl::last),
            napi::Property::new(env, "list")?.with_method(dsl::list),
            napi::Property::new(env, "lowerBound")?.with_method(dsl::lower_bound),
            napi::Property::new(env, "ltEq")?.with_method(dsl::lt_eq),
            napi::Property::new(env, "lt")?.with_method(dsl::lt),
            napi::Property::new(env, "max")?.with_method(dsl::max),
            napi::Property::new(env, "mean")?.with_method(dsl::mean),
            napi::Property::new(env, "median")?.with_method(dsl::median),
            napi::Property::new(env, "min")?.with_method(dsl::min),
            napi::Property::new(env, "mode")?.with_method(dsl::mode),
            napi::Property::new(env, "nUnique")?.with_method(dsl::n_unique),
            napi::Property::new(env, "neq")?.with_method(dsl::neq),
            napi::Property::new(env, "not")?.with_method(dsl::not),
            napi::Property::new(env, "or")?.with_method(dsl::or),
            napi::Property::new(env, "over")?.with_method(dsl::over),
            napi::Property::new(env, "pow")?.with_method(dsl::pow),
            napi::Property::new(env, "prefix")?.with_method(dsl::prefix),
            napi::Property::new(env, "quantile")?.with_method(dsl::quantile),
            napi::Property::new(env, "rank")?.with_method(dsl::rank),
            napi::Property::new(env, "reinterpret")?.with_method(dsl::reinterpret),
            napi::Property::new(env, "repeatBy")?.with_method(dsl::repeat_by),
            napi::Property::new(env, "reshape")?.with_method(dsl::reshape),
            napi::Property::new(env, "reverse")?.with_method(dsl::reverse),
            napi::Property::new(env, "rollingMax")?.with_method(dsl::rolling_max),
            napi::Property::new(env, "rollingMin")?.with_method(dsl::rolling_min),
            napi::Property::new(env, "rollingMean")?.with_method(dsl::rolling_mean),
            napi::Property::new(env, "rollingStd")?.with_method(dsl::rolling_std),
            napi::Property::new(env, "rollingSum")?.with_method(dsl::rolling_sum),
            napi::Property::new(env, "rollingVar")?.with_method(dsl::rolling_var),
            napi::Property::new(env, "rollingMedian")?.with_method(dsl::rolling_median),
            napi::Property::new(env, "rollingQuantile")?.with_method(dsl::rolling_quantile),
            napi::Property::new(env, "rollingSkew")?.with_method(dsl::rolling_skew),
            napi::Property::new(env, "round")?.with_method(dsl::round),
            napi::Property::new(env, "shiftAndFill")?.with_method(dsl::shift_and_fill),
            napi::Property::new(env, "shift")?.with_method(dsl::shift),
            napi::Property::new(env, "skew")?.with_method(dsl::skew),
            napi::Property::new(env, "slice")?.with_method(dsl::slice),
            napi::Property::new(env, "sort")?.with_method(dsl::sort),
            napi::Property::new(env, "sortBy")?.with_method(dsl::sort_by),
            napi::Property::new(env, "sortWith")?.with_method(dsl::sort_with),
            napi::Property::new(env, "std")?.with_method(dsl::std),
            napi::Property::new(env, "suffix")?.with_method(dsl::suffix),

            napi::Property::new(env, "sum")?.with_method(dsl::sum),
            napi::Property::new(env, "tail")?.with_method(dsl::tail),
            napi::Property::new(env, "takeEvery")?.with_method(dsl::take_every),
            napi::Property::new(env, "take")?.with_method(dsl::take),
            napi::Property::new(env, "unique")?.with_method(dsl::unique),
            napi::Property::new(env, "unique_stable")?.with_method(dsl::unique_stable),
            napi::Property::new(env, "upperBound")?.with_method(dsl::upper_bound),
            napi::Property::new(env, "var")?.with_method(dsl::var),
            napi::Property::new(env, "xor")?.with_method(dsl::xor),
        ])?;

        Ok(expr)
    }
}
