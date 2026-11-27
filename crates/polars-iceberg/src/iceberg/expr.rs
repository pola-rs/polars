//! Row filters for pruning, in the Iceberg REST expression JSON format (as serialized by
//! PyIceberg's `BooleanExpression.model_dump_json()`), bound to a table schema.
//!
//! Filters are only used to skip manifests and files; the engine always applies the full
//! predicate afterwards. Anything that cannot be bound or converted therefore becomes `True`
//! (keep everything), never an error.
use std::cmp::Ordering;

use chrono::{DateTime, NaiveDate, NaiveDateTime, NaiveTime, Timelike};
use serde_json::Value as JsonValue;

use crate::iceberg::spec::{NestedField, PrimitiveType, Schema, Type};
use crate::iceberg::values::{parse_decimal, parse_uuid};

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Op {
    IsNull,
    NotNull,
    IsNan,
    NotNan,
    Lt,
    LtEq,
    Gt,
    GtEq,
    Eq,
    NotEq,
    StartsWith,
    NotStartsWith,
    In,
    NotIn,
}

impl Op {
    fn negate(self) -> Self {
        use Op::*;
        match self {
            IsNull => NotNull,
            NotNull => IsNull,
            IsNan => NotNan,
            NotNan => IsNan,
            Lt => GtEq,
            LtEq => Gt,
            Gt => LtEq,
            GtEq => Lt,
            Eq => NotEq,
            NotEq => Eq,
            StartsWith => NotStartsWith,
            NotStartsWith => StartsWith,
            In => NotIn,
            NotIn => In,
        }
    }
}

/// A literal converted to the type of the field it is compared with.
#[derive(Debug, Clone, PartialEq)]
pub enum Lit {
    Bool(bool),
    /// int, long, date (days), time (us), timestamp (us or ns, per type).
    Int(i64),
    Float(f64),
    /// Unscaled value at the field's scale.
    Decimal(i128),
    Str(String),
    Bytes(Vec<u8>),
}

impl Lit {
    /// `None` if not comparable (different kinds, or NaN).
    pub fn partial_cmp(&self, other: &Lit) -> Option<Ordering> {
        match (self, other) {
            (Lit::Bool(a), Lit::Bool(b)) => Some(a.cmp(b)),
            (Lit::Int(a), Lit::Int(b)) => Some(a.cmp(b)),
            (Lit::Float(a), Lit::Float(b)) => a.partial_cmp(b),
            (Lit::Decimal(a), Lit::Decimal(b)) => Some(a.cmp(b)),
            (Lit::Str(a), Lit::Str(b)) => Some(a.as_bytes().cmp(b.as_bytes())),
            (Lit::Bytes(a), Lit::Bytes(b)) => Some(a.cmp(b)),
            _ => None,
        }
    }

    pub fn is_nan(&self) -> bool {
        matches!(self, Lit::Float(f) if f.is_nan())
    }
}

#[derive(Debug, Clone)]
pub struct Predicate {
    pub field_id: i32,
    pub ty: PrimitiveType,
    pub op: Op,
    /// One literal for comparisons, the set for `In` / `NotIn`, none for unary predicates.
    pub lits: Vec<Lit>,
}

#[derive(Debug, Clone)]
pub enum Bound {
    True,
    False,
    And(Box<Bound>, Box<Bound>),
    Or(Box<Bound>, Box<Bound>),
    Pred(Predicate),
}

impl Bound {
    /// The filter's structure without its literals, which may be sensitive, for logging.
    pub fn redacted(&self) -> String {
        match self {
            Bound::True => "true".into(),
            Bound::False => "false".into(),
            Bound::And(a, b) => format!("and({}, {})", a.redacted(), b.redacted()),
            Bound::Or(a, b) => format!("or({}, {})", a.redacted(), b.redacted()),
            Bound::Pred(p) => format!(
                "{:?}(field {}, {} literal(s))",
                p.op,
                p.field_id,
                p.lits.len()
            ),
        }
    }

    pub fn and(a: Bound, b: Bound) -> Bound {
        match (a, b) {
            (Bound::False, _) | (_, Bound::False) => Bound::False,
            (Bound::True, x) | (x, Bound::True) => x,
            (a, b) => Bound::And(Box::new(a), Box::new(b)),
        }
    }

    pub fn or(a: Bound, b: Bound) -> Bound {
        match (a, b) {
            (Bound::True, _) | (_, Bound::True) => Bound::True,
            (Bound::False, x) | (x, Bound::False) => x,
            (a, b) => Bound::Or(Box::new(a), Box::new(b)),
        }
    }

    /// Field IDs referenced by predicates.
    pub fn field_ids(&self, out: &mut Vec<i32>) {
        match self {
            Bound::True | Bound::False => {},
            Bound::And(a, b) | Bound::Or(a, b) => {
                a.field_ids(out);
                b.field_ids(out);
            },
            Bound::Pred(p) => out.push(p.field_id),
        }
    }

    /// Evaluate with a three-valued predicate evaluator, where `true` means "rows might match".
    pub fn eval(&self, f: &mut impl FnMut(&Predicate) -> bool) -> bool {
        match self {
            Bound::True => true,
            Bound::False => false,
            Bound::And(a, b) => a.eval(f) && b.eval(f),
            Bound::Or(a, b) => a.eval(f) || b.eval(f),
            Bound::Pred(p) => f(p),
        }
    }
}

/// Parse and bind a filter to `schema`. `NOT` is pushed down to the predicates. Parts that
/// cannot be parsed, bound or converted become `True`.
pub fn bind(json: &JsonValue, schema: &Schema) -> Bound {
    bind_node(json, schema, false)
}

/// Binds `json`, negated if `negated`. `NOT` is applied while binding, before unbindable parts
/// become `True` (negating an already folded `True` would prune).
fn bind_node(json: &JsonValue, schema: &Schema, negated: bool) -> Bound {
    bind_impl(json, schema, negated).unwrap_or(Bound::True)
}

fn bind_impl(json: &JsonValue, schema: &Schema, negated: bool) -> Option<Bound> {
    let constant = |v: bool| {
        Some(if v != negated {
            Bound::True
        } else {
            Bound::False
        })
    };

    let obj = match json {
        JsonValue::Bool(v) => return constant(*v),
        JsonValue::String(s) if s == "true" => return constant(true),
        JsonValue::String(s) if s == "false" => return constant(false),
        JsonValue::Object(obj) => obj,
        _ => return None,
    };

    let ty = obj.get("type")?.as_str()?;

    let child = |key: &str| Some(bind_node(obj.get(key)?, schema, negated));

    let op = match ty {
        "true" => return constant(true),
        "false" => return constant(false),
        // De Morgan: under `NOT`, AND and OR swap.
        "and" | "or" => {
            let (l, r) = (child("left")?, child("right")?);
            return Some(if (ty == "and") != negated {
                Bound::and(l, r)
            } else {
                Bound::or(l, r)
            });
        },
        "not" => return bind_impl(obj.get("child")?, schema, !negated),
        "is-null" => Op::IsNull,
        "not-null" => Op::NotNull,
        "is-nan" => Op::IsNan,
        "not-nan" => Op::NotNan,
        "lt" => Op::Lt,
        "lt-eq" => Op::LtEq,
        "gt" => Op::Gt,
        "gt-eq" => Op::GtEq,
        "eq" => Op::Eq,
        "not-eq" => Op::NotEq,
        "starts-with" => Op::StartsWith,
        "not-starts-with" => Op::NotStartsWith,
        "in" => Op::In,
        "not-in" => Op::NotIn,
        _ => return None,
    };

    // Negated before binding: binding may widen literals (see `literal_range`), and a widened
    // predicate cannot be negated.
    bind_predicate(obj, if negated { op.negate() } else { op }, schema)
}

/// A predicate bound to `schema`.
fn bind_predicate(
    obj: &serde_json::Map<String, JsonValue>,
    op: Op,
    schema: &Schema,
) -> Option<Bound> {
    let (field, required) = resolve_term(obj.get("term")?, schema)?;
    let Type::Primitive(pt) = &field.field_type else {
        return None;
    };

    // Unary predicates that cannot hold for the type (e.g. NaN checks on non-float columns).
    let is_float = matches!(pt, PrimitiveType::Float | PrimitiveType::Double);
    match op {
        Op::IsNan if !is_float => return Some(Bound::False),
        Op::NotNan if !is_float => return Some(Bound::True),
        // A required field added without an `initial-default` is null in older files (both
        // Java and PyIceberg allow adding one), so only a default guarantees values.
        Op::IsNull if required && field.initial_default.is_some() => {
            return Some(Bound::False);
        },
        Op::NotNull if required => return Some(Bound::True),
        _ => {},
    }

    let ranges = match op {
        Op::IsNull | Op::NotNull | Op::IsNan | Op::NotNan => vec![],
        Op::In | Op::NotIn => obj
            .get("values")?
            .as_array()?
            .iter()
            .map(|v| literal_range(v, pt))
            .collect::<Option<Vec<_>>>()?,
        _ => vec![literal_range(obj.get("value")?, pt)?],
    };

    let pred = |op: Op, lits: Vec<Lit>| {
        Bound::Pred(Predicate {
            field_id: field.id,
            ty: pt.clone(),
            op,
            lits,
        })
    };

    if ranges.iter().all(|(lo, hi)| lo == hi) {
        return Some(pred(op, ranges.into_iter().map(|(lo, _)| lo).collect()));
    }

    // Some literal stands for any value in `lo..=hi`: keep what any of them might match.
    Some(match op {
        Op::Lt | Op::LtEq => pred(Op::LtEq, vec![ranges[0].1.clone()]),
        Op::Gt | Op::GtEq => pred(op, vec![ranges[0].0.clone()]),
        Op::Eq | Op::In => ranges.into_iter().fold(Bound::False, |acc, (lo, hi)| {
            Bound::or(
                acc,
                Bound::and(pred(Op::GtEq, vec![lo]), pred(Op::LtEq, vec![hi])),
            )
        }),
        _ => Bound::True,
    })
}

/// The range of values a JSON literal may stand for, converted to `ty`. Exact, except for
/// nanosecond timestamps given with at most microsecond precision: Polars builds datetime
/// literals through Python `datetime`, which floors nanoseconds to microseconds
/// (`to_py_datetime`), so up to 999 ns may have been dropped.
fn literal_range(v: &JsonValue, ty: &PrimitiveType) -> Option<(Lit, Lit)> {
    let lit = convert_literal(v, ty)?;
    if let (PrimitiveType::TimestampNs | PrimitiveType::TimestamptzNs, JsonValue::String(s)) =
        (ty, v)
        && let Lit::Int(ns) = lit
    {
        let fraction_digits = s
            .split_once('.')
            .map_or(0, |(_, f)| f.bytes().take_while(u8::is_ascii_digit).count());
        if fraction_digits <= 6 {
            return Some((lit, Lit::Int(ns.checked_add(999)?)));
        }
    }
    Some((lit.clone(), lit))
}

/// A term is a (possibly dotted) column name, or `{"type": "reference", "term": name}`. Also
/// returns whether the field is never null: it and all its parent structs are required.
fn resolve_term<'a>(term: &JsonValue, schema: &'a Schema) -> Option<(&'a NestedField, bool)> {
    let name = match term {
        JsonValue::String(s) => s.as_str(),
        JsonValue::Object(obj) if obj.get("type").and_then(|t| t.as_str()) == Some("reference") => {
            obj.get("term")?.as_str()?
        },
        _ => return None,
    };

    let mut fields = &schema.fields;
    let mut required = true;
    let mut parts = name.split('.').peekable();
    loop {
        let part = parts.next()?;
        let field = fields.iter().find(|f| f.name == part)?;
        required &= field.required;
        if parts.peek().is_none() {
            return Some((field, required));
        }
        match &field.field_type {
            Type::Struct(children) => fields = children,
            _ => return None,
        }
    }
}

/// Convert a JSON literal to a field type, following PyIceberg's literal conversions. `None`
/// if it cannot be represented exactly.
pub fn convert_literal(v: &JsonValue, ty: &PrimitiveType) -> Option<Lit> {
    use PrimitiveType as P;
    match (v, ty) {
        (JsonValue::Bool(b), P::Boolean) => Some(Lit::Bool(*b)),
        (JsonValue::Number(n), P::Int) => {
            let i = n.as_i64()?;
            i32::try_from(i).ok().map(|i| Lit::Int(i64::from(i)))
        },
        (JsonValue::Number(n), P::Long | P::Date | P::Time | P::Timestamp | P::Timestamptz) => {
            n.as_i64().map(Lit::Int)
        },
        (JsonValue::Number(n), P::Float | P::Double) => n.as_f64().map(Lit::Float),
        (JsonValue::Number(n), P::Decimal { scale, .. }) => {
            parse_decimal(&n.to_string(), *scale).map(Lit::Decimal)
        },
        (JsonValue::String(s), P::String) => Some(Lit::Str(s.clone())),
        (JsonValue::String(s), P::Date) => {
            let d = NaiveDate::parse_from_str(s, "%Y-%m-%d").ok()?;
            let epoch = NaiveDate::from_ymd_opt(1970, 1, 1)?;
            Some(Lit::Int((d - epoch).num_days()))
        },
        (JsonValue::String(s), P::Time) => {
            let t = NaiveTime::parse_from_str(s, "%H:%M:%S%.f").ok()?;
            // Flooring sub-microsecond digits would narrow e.g. `<`.
            if t.nanosecond() % 1000 != 0 {
                return None;
            }
            Some(Lit::Int(
                i64::from(t.num_seconds_from_midnight()) * 1_000_000
                    + i64::from(t.nanosecond()) / 1000,
            ))
        },
        (JsonValue::String(s), P::Timestamp | P::TimestampNs) => {
            // Timezone-aware strings are rejected for timestamps without timezone.
            let ts = NaiveDateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f")
                .ok()?
                .and_utc();
            if matches!(ty, P::Timestamp) && ts.timestamp_subsec_nanos() % 1000 != 0 {
                return None;
            }
            Some(Lit::Int(if matches!(ty, P::Timestamp) {
                ts.timestamp_micros()
            } else {
                ts.timestamp_nanos_opt()?
            }))
        },
        (JsonValue::String(s), P::Timestamptz | P::TimestamptzNs) => {
            let ts = DateTime::parse_from_rfc3339(s)
                .or_else(|_| DateTime::parse_from_str(s, "%Y-%m-%dT%H:%M:%S%.f%:z"))
                .ok()?;
            if matches!(ty, P::Timestamptz) && ts.timestamp_subsec_nanos() % 1000 != 0 {
                return None;
            }
            Some(Lit::Int(if matches!(ty, P::Timestamptz) {
                ts.timestamp_micros()
            } else {
                ts.timestamp_nanos_opt()?
            }))
        },
        (JsonValue::String(s), P::Decimal { scale, .. }) => {
            parse_decimal(s, *scale).map(Lit::Decimal)
        },
        (JsonValue::String(s), P::Uuid) => parse_uuid(s).map(Lit::Bytes),
        _ => None,
    }
}

#[cfg(test)]
mod tests {
    use serde_json::json;

    use super::*;

    fn schema() -> Schema {
        let field = |id, name: &str, ty| NestedField {
            id,
            name: name.into(),
            required: false,
            field_type: Type::Primitive(ty),
            initial_default: None,
        };
        Schema::new(
            0,
            vec![
                field(1, "ts_ns", PrimitiveType::TimestampNs),
                field(2, "ts_us", PrimitiveType::Timestamp),
            ],
        )
    }

    fn pred(bound: &Bound) -> (Op, &[Lit]) {
        match bound {
            Bound::Pred(p) => (p.op, &p.lits),
            other => panic!("expected a predicate, got {other:?}"),
        }
    }

    #[test]
    fn null_checks_on_required_field_of_optional_struct_are_kept() {
        let leaf = NestedField {
            id: 2,
            name: "x".into(),
            required: true,
            field_type: Type::Primitive(PrimitiveType::Long),
            initial_default: None,
        };
        let schema = Schema::new(
            0,
            vec![NestedField {
                id: 1,
                name: "s".into(),
                required: false,
                field_type: Type::Struct(vec![leaf]),
                initial_default: None,
            }],
        );
        // `s.x` is null where `s` is.
        let bound = bind(&json!({"type": "is-null", "term": "s.x"}), &schema);
        assert_eq!(pred(&bound).0, Op::IsNull);
        let bound = bind(&json!({"type": "not-null", "term": "s.x"}), &schema);
        assert_eq!(pred(&bound).0, Op::NotNull);
    }

    #[test]
    fn is_null_on_required_field_is_kept_without_initial_default() {
        // A required field added without a default is null in older files.
        let schema = |initial_default| {
            Schema::new(
                0,
                vec![NestedField {
                    id: 1,
                    name: "r".into(),
                    required: true,
                    field_type: Type::Primitive(PrimitiveType::Long),
                    initial_default,
                }],
            )
        };
        let is_null = json!({"type": "is-null", "term": "r"});
        assert_eq!(pred(&bind(&is_null, &schema(None))).0, Op::IsNull);
        assert!(matches!(
            bind(&is_null, &schema(Some(json!(1)))),
            Bound::False
        ));
    }

    #[test]
    fn redacted_filter_has_no_literals() {
        let bound = bind(
            &json!({"type": "eq", "term": "ts_us", "value": "2020-01-01T00:00:00.123456"}),
            &schema(),
        );
        let text = bound.redacted();
        assert!(!text.contains("123456") && !text.contains("2020"), "{text}");
        assert!(text.contains("field 2"), "{text}");
    }

    // 2020-01-01T00:00:00.000001 in nanoseconds.
    const LO: i64 = 1_577_836_800_000_001_000;
    const HI: i64 = LO + 999;

    #[test]
    fn microsecond_literal_on_nanosecond_column_is_widened() {
        let schema = schema();
        let bind = |op: &str| {
            bind(
                &json!({"type": op, "term": "ts_ns", "value": "2020-01-01T00:00:00.000001"}),
                &schema,
            )
        };

        assert_eq!(pred(&bind("lt")), (Op::LtEq, &[Lit::Int(HI)][..]));
        assert_eq!(pred(&bind("lt-eq")), (Op::LtEq, &[Lit::Int(HI)][..]));
        assert_eq!(pred(&bind("gt")), (Op::Gt, &[Lit::Int(LO)][..]));
        assert_eq!(pred(&bind("gt-eq")), (Op::GtEq, &[Lit::Int(LO)][..]));
        assert!(matches!(bind("not-eq"), Bound::True));

        let Bound::And(lo, hi) = bind("eq") else {
            panic!("expected a range");
        };
        assert_eq!(pred(&lo), (Op::GtEq, &[Lit::Int(LO)][..]));
        assert_eq!(pred(&hi), (Op::LtEq, &[Lit::Int(HI)][..]));
    }

    #[test]
    fn widened_literal_is_negated_before_widening() {
        let schema = schema();
        // `NOT (x < L)` is `x >= L`, so the lower end of the range applies.
        let bound = bind(
            &json!({"type": "not", "child":
                {"type": "lt", "term": "ts_ns", "value": "2020-01-01T00:00:00.000001"}}),
            &schema,
        );
        assert_eq!(pred(&bound), (Op::GtEq, &[Lit::Int(LO)][..]));
    }

    #[test]
    fn exact_literals_are_not_widened() {
        let schema = schema();
        let bound = bind(
            &json!({"type": "lt", "term": "ts_ns", "value": "2020-01-01T00:00:00.000001000"}),
            &schema,
        );
        assert_eq!(pred(&bound), (Op::Lt, &[Lit::Int(LO)][..]));

        let bound = bind(
            &json!({"type": "eq", "term": "ts_us", "value": "2020-01-01T00:00:00.000001"}),
            &schema,
        );
        assert_eq!(
            pred(&bound),
            (Op::Eq, &[Lit::Int(1_577_836_800_000_001)][..])
        );
    }

    #[test]
    fn sub_microsecond_literals_are_not_floored() {
        let convert = |v: &str, ty| convert_literal(&json!(v), &ty);
        assert_eq!(
            convert("00:00:00.000001", PrimitiveType::Time),
            Some(Lit::Int(1))
        );
        assert_eq!(convert("00:00:00.0000015", PrimitiveType::Time), None);
        assert_eq!(
            convert("1970-01-01T00:00:00.000001000", PrimitiveType::Timestamp),
            Some(Lit::Int(1))
        );
        assert_eq!(
            convert("1970-01-01T00:00:00.0000015", PrimitiveType::Timestamp),
            None
        );
        assert_eq!(
            convert(
                "1970-01-01T00:00:00.0000015+00:00",
                PrimitiveType::Timestamptz
            ),
            None
        );
        // Nanosecond columns keep the digits.
        assert_eq!(
            convert("1970-01-01T00:00:00.0000015", PrimitiveType::TimestampNs),
            Some(Lit::Int(1500))
        );

        // `x < 1.5 us` must not become `x < 1 us`.
        let bound = bind(
            &json!({"type": "lt", "term": "ts_us", "value": "2020-01-01T00:00:00.0000015"}),
            &schema(),
        );
        assert!(matches!(bound, Bound::True));
    }

    #[test]
    fn float_literals_parse_exactly() {
        // Without serde_json's `float_roundtrip`, these parse one ULP off.
        for (text, expected) in [
            ("123456789.98765433", 123456789.98765433_f64),
            (
                "339999995214436424907732413799364296704",
                3.3999999521443642e38,
            ),
        ] {
            let v: JsonValue = serde_json::from_str(text).unwrap();
            assert_eq!(
                convert_literal(&v, &PrimitiveType::Double),
                Some(Lit::Float(expected))
            );
        }
    }
}
