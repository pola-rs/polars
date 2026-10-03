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

    fn negate(self) -> Bound {
        match self {
            Bound::True => Bound::False,
            Bound::False => Bound::True,
            Bound::And(a, b) => Bound::or(a.negate(), b.negate()),
            Bound::Or(a, b) => Bound::and(a.negate(), b.negate()),
            Bound::Pred(mut p) => {
                p.op = p.op.negate();
                Bound::Pred(p)
            },
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

    let pred = bind_predicate(obj, op, schema)?;
    Some(if negated { pred.negate() } else { pred })
}

/// A predicate bound to `schema` (contains no unbindable parts, so it is safe to negate).
fn bind_predicate(
    obj: &serde_json::Map<String, JsonValue>,
    op: Op,
    schema: &Schema,
) -> Option<Bound> {
    let field = resolve_term(obj.get("term")?, schema)?;
    let Type::Primitive(pt) = &field.field_type else {
        return None;
    };

    let lits = match op {
        Op::IsNull | Op::NotNull | Op::IsNan | Op::NotNan => vec![],
        Op::In | Op::NotIn => obj
            .get("values")?
            .as_array()?
            .iter()
            .map(|v| convert_literal(v, pt))
            .collect::<Option<Vec<_>>>()?,
        _ => vec![convert_literal(obj.get("value")?, pt)?],
    };

    // Unary predicates that cannot hold for the type (e.g. NaN checks on non-float columns).
    let is_float = matches!(pt, PrimitiveType::Float | PrimitiveType::Double);
    match op {
        Op::IsNan if !is_float => return Some(Bound::False),
        Op::NotNan if !is_float => return Some(Bound::True),
        Op::IsNull if field.required => return Some(Bound::False),
        Op::NotNull if field.required => return Some(Bound::True),
        _ => {},
    }

    Some(Bound::Pred(Predicate {
        field_id: field.id,
        ty: pt.clone(),
        op,
        lits,
    }))
}

/// A term is a (possibly dotted) column name, or `{"type": "reference", "term": name}`.
fn resolve_term<'a>(term: &JsonValue, schema: &'a Schema) -> Option<&'a NestedField> {
    let name = match term {
        JsonValue::String(s) => s.as_str(),
        JsonValue::Object(obj) if obj.get("type").and_then(|t| t.as_str()) == Some("reference") => {
            obj.get("term")?.as_str()?
        },
        _ => return None,
    };

    let mut fields = &schema.fields;
    let mut parts = name.split('.').peekable();
    loop {
        let part = parts.next()?;
        let field = fields.iter().find(|f| f.name == part)?;
        if parts.peek().is_none() {
            return Some(field);
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
            Some(Lit::Int(if matches!(ty, P::Timestamptz) {
                ts.timestamp_micros()
            } else {
                ts.timestamp_nanos_opt()?
            }))
        },
        (JsonValue::String(s), P::Decimal { scale, .. }) => {
            parse_decimal(s, *scale).map(Lit::Decimal)
        },
        (JsonValue::String(s), P::Uuid) => {
            let hex_str: String = s.chars().filter(|c| *c != '-').collect();
            let bytes = hex::decode(hex_str).ok()?;
            (bytes.len() == 16).then_some(Lit::Bytes(bytes))
        },
        _ => None,
    }
}

/// Parse a decimal string into an unscaled integer; `None` if it has more digits than `scale`
/// after the decimal point.
fn parse_decimal(s: &str, scale: u32) -> Option<i128> {
    let (negative, digits) = match s.strip_prefix('-') {
        Some(rest) => (true, rest),
        None => (false, s.strip_prefix('+').unwrap_or(s)),
    };
    let (int_part, frac_part) = digits.split_once('.').unwrap_or((digits, ""));
    let frac_part = frac_part.trim_end_matches('0');
    if frac_part.len() > scale as usize
        || int_part.is_empty() && frac_part.is_empty()
        || !int_part
            .chars()
            .chain(frac_part.chars())
            .all(|c| c.is_ascii_digit())
    {
        return None;
    }
    let mut value: i128 = if int_part.is_empty() {
        0
    } else {
        int_part.parse().ok()?
    };
    for i in 0..scale as usize {
        let digit = frac_part
            .as_bytes()
            .get(i)
            .map_or(0, |b| i128::from(b - b'0'));
        value = value.checked_mul(10)?.checked_add(digit)?;
    }
    Some(if negative { -value } else { value })
}

#[cfg(test)]
mod tests {
    use super::*;

    fn schema() -> Schema {
        Schema::new(
            0,
            vec![
                NestedField {
                    id: 1,
                    name: "a".into(),
                    required: false,
                    field_type: Type::Primitive(PrimitiveType::Long),
                    initial_default: None,
                },
                NestedField {
                    id: 2,
                    name: "d".into(),
                    required: false,
                    field_type: Type::Primitive(PrimitiveType::Date),
                    initial_default: None,
                },
            ],
        )
    }

    #[test]
    fn test_bind() {
        let json: JsonValue = serde_json::from_str(
            r#"{"type":"not","child":{"type":"and","left":{"term":"a","type":"gt","value":1},
            "right":{"term":"d","type":"lt","value":"1970-01-03"}}}"#,
        )
        .unwrap();
        let bound = bind(&json, &schema());
        let Bound::Or(l, r) = bound else {
            panic!("{bound:?}")
        };
        let (Bound::Pred(l), Bound::Pred(r)) = (*l, *r) else {
            panic!()
        };
        assert_eq!((l.op, &l.lits[..]), (Op::LtEq, &[Lit::Int(1)][..]));
        assert_eq!((r.op, &r.lits[..]), (Op::GtEq, &[Lit::Int(2)][..]));

        // Unknown columns do not prune, even under NOT.
        let json: JsonValue =
            serde_json::from_str(r#"{"type":"not","child":{"term":"x","type":"is-null"}}"#)
                .unwrap();
        assert!(matches!(bind(&json, &schema()), Bound::True));

        // Unbindable leaves inside AND / OR under NOT stay `True` instead of being negated.
        // `a == 1.5` (long column) cannot be bound: `NOT (a == 1.5 OR a > 5)` is
        // `True AND a <= 5`.
        let json: JsonValue = serde_json::from_str(
            r#"{"type":"not","child":{"type":"or","left":{"term":"a","type":"eq","value":1.5},
            "right":{"term":"a","type":"gt","value":5}}}"#,
        )
        .unwrap();
        let Bound::Pred(p) = bind(&json, &schema()) else {
            panic!()
        };
        assert_eq!((p.op, &p.lits[..]), (Op::LtEq, &[Lit::Int(5)][..]));

        // `NOT (a == 1.5 AND a > 5)` is `True OR a <= 5`.
        let json: JsonValue = serde_json::from_str(
            r#"{"type":"not","child":{"type":"and","left":{"term":"a","type":"eq","value":1.5},
            "right":{"term":"a","type":"gt","value":5}}}"#,
        )
        .unwrap();
        assert!(matches!(bind(&json, &schema()), Bound::True));

        // Double negation.
        let json: JsonValue = serde_json::from_str(
            r#"{"type":"not","child":{"type":"not","child":{"term":"a","type":"gt","value":5}}}"#,
        )
        .unwrap();
        let Bound::Pred(p) = bind(&json, &schema()) else {
            panic!()
        };
        assert_eq!(p.op, Op::Gt);
    }

    #[test]
    fn test_parse_decimal() {
        assert_eq!(parse_decimal("1.00", 2), Some(100));
        assert_eq!(parse_decimal("1.50", 1), Some(15));
        assert_eq!(parse_decimal("1.25", 1), None);
    }
}
