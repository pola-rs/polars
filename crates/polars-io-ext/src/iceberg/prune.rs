//! Manifest and file pruning with a bound row filter ([`crate::iceberg::expr`]).
//!
//! Follows PyIceberg's planning evaluators (inclusive projection onto partition specs, the
//! manifest evaluator over partition summaries, partition value evaluation, and the inclusive
//! metrics evaluator), with one deliberate difference: NaN. Polars orders NaN above every other
//! value (`NaN > x` is true), whereas Iceberg evaluators treat comparisons with NaN as false. So
//! greater-than predicates on float columns only prune when NaNs are known to be absent.
//!
//! Every function answers "might rows match?"; `true` keeps the manifest / file.
use std::cmp::Ordering;

use polars_utils::aliases::PlHashMap;

use crate::iceberg::avro::Datum;
use crate::iceberg::expr::{Bound, Lit, Op, Predicate};
use crate::iceberg::manifest::{DataFile, FieldSummary, ManifestFile};
use crate::iceberg::spec::{PartitionSpec, PrimitiveType, Schema, Transform, Type};

const MICROS_PER_HOUR: i64 = 3_600_000_000;
const MICROS_PER_DAY: i64 = 86_400_000_000;

pub struct Pruner {
    filter: Bound,
    /// Spec ID → the filter projected onto that spec's partition fields (predicate field IDs
    /// are indices into the partition tuple).
    projections: PlHashMap<i32, SpecProjection>,
}

struct SpecProjection {
    filter: Bound,
}

impl Pruner {
    pub fn new(filter: Bound, specs: &PlHashMap<i32, std::sync::Arc<PartitionSpec>>) -> Self {
        let projections = specs
            .iter()
            .map(|(id, spec)| {
                (
                    *id,
                    SpecProjection {
                        filter: project(&filter, spec),
                    },
                )
            })
            .collect();
        Self {
            filter,
            projections,
        }
    }

    /// Field IDs whose column metrics the pruner uses.
    pub fn field_ids(&self) -> Vec<i32> {
        let mut out = vec![];
        self.filter.field_ids(&mut out);
        out
    }

    pub fn is_trivial(&self) -> bool {
        matches!(self.filter, Bound::True)
    }

    /// Whether a manifest might contain matching data files, from its partition summaries.
    pub fn manifest_might_match(
        &self,
        manifest: &ManifestFile,
        spec: &PartitionSpec,
        schema: &Schema,
    ) -> bool {
        let Some(projection) = self.projections.get(&manifest.spec_id) else {
            return true;
        };
        projection.filter.eval(&mut |p| {
            let Some(summary) = manifest.partitions.get(p.field_id as usize) else {
                return true;
            };
            let Some(ty) = partition_type(spec, p.field_id as usize, schema) else {
                return true;
            };
            summary_might_match(p, summary, &ty)
        })
    }

    /// Whether a data file might contain matching rows, from its partition tuple (in spec
    /// order) and column metrics.
    pub fn file_might_match(
        &self,
        spec_id: i32,
        spec: &PartitionSpec,
        schema: &Schema,
        partition: &[Option<Datum>],
        file: &DataFile,
    ) -> bool {
        if let Some(projection) = self.projections.get(&spec_id) {
            let partition_match = projection.filter.eval(&mut |p| {
                let idx = p.field_id as usize;
                let Some(ty) = partition_type(spec, idx, schema) else {
                    return true;
                };
                match partition.get(idx) {
                    Some(value) => {
                        let value = match value {
                            None => None,
                            Some(d) => match datum_to_lit(d, &ty) {
                                Some(v) => Some(v),
                                None => return true,
                            },
                        };
                        value_might_match(p, value.as_ref())
                    },
                    None => true,
                }
            });
            if !partition_match {
                return false;
            }
        }

        metrics_might_match(&self.filter, file)
    }
}

/// Result type of a partition field.
fn partition_type(spec: &PartitionSpec, idx: usize, schema: &Schema) -> Option<PrimitiveType> {
    let field = spec.fields.get(idx)?;
    let source = schema.field_by_id(field.source_id)?;
    let Type::Primitive(source_type) = &source.field_type else {
        return None;
    };
    match field.transform {
        Transform::Identity | Transform::Truncate(_) => Some(source_type.clone()),
        Transform::Year
        | Transform::Month
        | Transform::Day
        | Transform::Hour
        | Transform::Bucket(_) => Some(PrimitiveType::Int),
        Transform::Void | Transform::Other(_) => None,
    }
}

// Inclusive projection.

fn project(filter: &Bound, spec: &PartitionSpec) -> Bound {
    match filter {
        Bound::True => Bound::True,
        Bound::False => Bound::False,
        Bound::And(a, b) => Bound::and(project(a, spec), project(b, spec)),
        Bound::Or(a, b) => Bound::or(project(a, spec), project(b, spec)),
        Bound::Pred(p) => spec
            .fields
            .iter()
            .enumerate()
            .filter(|(_, f)| f.source_id == p.field_id)
            .map(|(i, f)| project_predicate(p, i, &f.transform))
            .fold(Bound::True, Bound::and),
    }
}

fn project_predicate(p: &Predicate, idx: usize, transform: &Transform) -> Bound {
    let with = |op: Op, lits: Vec<Lit>, ty: PrimitiveType| {
        Bound::Pred(Predicate {
            field_id: idx as i32,
            ty,
            op,
            lits,
        })
    };

    // Null checks project through every value-preserving transform.
    if matches!(p.op, Op::IsNull | Op::NotNull) {
        return match transform {
            Transform::Void | Transform::Other(_) => Bound::True,
            Transform::Identity | Transform::Truncate(_) => with(p.op, vec![], p.ty.clone()),
            _ => with(p.op, vec![], PrimitiveType::Int),
        };
    }

    let apply_all =
        |f: &dyn Fn(&Lit) -> Option<Lit>| p.lits.iter().map(f).collect::<Option<Vec<_>>>();

    match transform {
        Transform::Identity => with(p.op, p.lits.clone(), p.ty.clone()),
        Transform::Year | Transform::Month | Transform::Day | Transform::Hour => {
            let t = |l: &Lit| temporal(transform, &p.ty, l);
            // Transforms are monotonic (floor), so `x < v` implies `t(x) <= t(v)`.
            let (op, lits) = match p.op {
                Op::Lt | Op::LtEq => (Op::LtEq, apply_all(&t)),
                Op::Gt | Op::GtEq => (Op::GtEq, apply_all(&t)),
                Op::Eq => (Op::Eq, apply_all(&t)),
                Op::In => (Op::In, apply_all(&t)),
                _ => return Bound::True,
            };
            match lits {
                Some(lits) => with(op, lits, PrimitiveType::Int),
                None => Bound::True,
            }
        },
        Transform::Truncate(width) => {
            let t = |l: &Lit| truncate(*width, l);
            let (op, lits) = match p.op {
                Op::Lt | Op::LtEq => (Op::LtEq, apply_all(&t)),
                Op::Gt | Op::GtEq => (Op::GtEq, apply_all(&t)),
                Op::Eq => (Op::Eq, apply_all(&t)),
                Op::In => (Op::In, apply_all(&t)),
                Op::StartsWith => match &p.lits[..] {
                    [Lit::Str(prefix)] if prefix.chars().count() >= *width as usize => {
                        (Op::Eq, apply_all(&t))
                    },
                    [Lit::Str(_)] => (Op::StartsWith, Some(p.lits.clone())),
                    _ => return Bound::True,
                },
                _ => return Bound::True,
            };
            match lits {
                Some(lits) => with(op, lits, p.ty.clone()),
                None => Bound::True,
            }
        },
        // Bucket values could be computed for `Eq` / `In` (murmur3); not implemented, so no
        // pruning.
        Transform::Bucket(_) | Transform::Void | Transform::Other(_) => Bound::True,
    }
}

fn temporal(transform: &Transform, ty: &PrimitiveType, lit: &Lit) -> Option<Lit> {
    use PrimitiveType as P;
    let Lit::Int(v) = lit else { return None };
    let micros = match ty {
        P::Date => None,
        P::Timestamp | P::Timestamptz => Some(*v),
        P::TimestampNs | P::TimestamptzNs => Some(v.div_euclid(1000)),
        _ => return None,
    };
    let days = match micros {
        Some(us) => us.div_euclid(MICROS_PER_DAY),
        None => *v,
    };
    let out = match transform {
        Transform::Hour => micros?.div_euclid(MICROS_PER_HOUR),
        Transform::Day => days,
        Transform::Month | Transform::Year => {
            use chrono::Datelike;
            let date = chrono::NaiveDate::from_ymd_opt(1970, 1, 1)?
                .checked_add_signed(chrono::Duration::days(days))?;
            let years = i64::from(date.year() - 1970);
            if matches!(transform, Transform::Year) {
                years
            } else {
                years * 12 + i64::from(date.month0())
            }
        },
        _ => return None,
    };
    Some(Lit::Int(out))
}

fn truncate(width: u32, lit: &Lit) -> Option<Lit> {
    let w = i64::from(width);
    match lit {
        Lit::Int(v) if w > 0 => Some(Lit::Int(v - v.rem_euclid(w))),
        Lit::Str(s) => Some(Lit::Str(s.chars().take(width as usize).collect())),
        Lit::Bytes(b) => Some(Lit::Bytes(b.iter().take(width as usize).copied().collect())),
        _ => None,
    }
}

// Evaluation of an exact value (partition tuples).

fn value_might_match(p: &Predicate, value: Option<&Lit>) -> bool {
    let Some(value) = value else {
        // Comparisons with null never hold.
        return p.op == Op::IsNull;
    };
    if value.is_nan() {
        // NaN ordering differs between Iceberg and Polars; see the module docs.
        return p.op != Op::IsNull && p.op != Op::NotNan;
    }
    let cmp = |lit: &Lit| value.partial_cmp(lit);
    let lit = p.lits.first();
    match p.op {
        Op::IsNull => false,
        Op::NotNull => true,
        Op::IsNan => false,
        Op::NotNan => true,
        Op::Lt => lit.and_then(cmp).is_none_or(|o| o == Ordering::Less),
        Op::LtEq => lit.and_then(cmp).is_none_or(|o| o != Ordering::Greater),
        Op::Gt => lit.and_then(cmp).is_none_or(|o| o == Ordering::Greater),
        Op::GtEq => lit.and_then(cmp).is_none_or(|o| o != Ordering::Less),
        Op::Eq => lit.and_then(cmp).is_none_or(|o| o == Ordering::Equal),
        Op::NotEq => lit.and_then(cmp).is_none_or(|o| o != Ordering::Equal),
        Op::In => p
            .lits
            .iter()
            .any(|l| cmp(l).is_none_or(|o| o == Ordering::Equal)),
        Op::NotIn => p
            .lits
            .iter()
            .all(|l| cmp(l).is_none_or(|o| o != Ordering::Equal)),
        Op::StartsWith | Op::NotStartsWith => match (value, lit) {
            (Lit::Str(v), Some(Lit::Str(prefix))) => {
                v.starts_with(prefix.as_str()) == (p.op == Op::StartsWith)
            },
            _ => true,
        },
    }
}

// Partition summaries.

fn summary_might_match(p: &Predicate, s: &FieldSummary, ty: &PrimitiveType) -> bool {
    let is_float = matches!(ty, PrimitiveType::Float | PrimitiveType::Double);
    let may_have_nan = is_float && s.contains_nan != Some(false);
    let lower = s.lower_bound.as_deref().and_then(|b| decode_bound(b, ty));
    let upper = s.upper_bound.as_deref().and_then(|b| decode_bound(b, ty));

    match p.op {
        Op::IsNull => s.contains_null,
        Op::NotNull => !(s.contains_null && s.lower_bound.is_none() && !may_have_nan),
        Op::IsNan => may_have_nan,
        Op::NotNan | Op::NotEq | Op::NotIn | Op::NotStartsWith => true,
        _ => {
            if s.lower_bound.is_none() && s.upper_bound.is_none() {
                // All values are null (and NaN, for floats).
                return may_have_nan && matches!(p.op, Op::Gt | Op::GtEq);
            }
            bounds_might_match(p, lower.as_ref(), upper.as_ref(), may_have_nan)
        },
    }
}

/// Range checks shared by summaries and column metrics. Bounds exclude nulls and NaN.
fn bounds_might_match(
    p: &Predicate,
    lower: Option<&Lit>,
    upper: Option<&Lit>,
    may_have_nan: bool,
) -> bool {
    if lower.is_some_and(Lit::is_nan) || upper.is_some_and(Lit::is_nan) {
        return true;
    }
    let lit = p.lits.first();
    let cmp = |bound: Option<&Lit>, lit: Option<&Lit>| match (bound, lit) {
        (Some(b), Some(l)) => b.partial_cmp(l),
        _ => None,
    };
    match p.op {
        // `lower >= v`: every value is >= v.
        Op::Lt => !cmp(lower, lit).is_some_and(|o| o != Ordering::Less),
        Op::LtEq => !cmp(lower, lit).is_some_and(|o| o == Ordering::Greater),
        Op::Gt => may_have_nan || !cmp(upper, lit).is_some_and(|o| o != Ordering::Greater),
        Op::GtEq => may_have_nan || !cmp(upper, lit).is_some_and(|o| o == Ordering::Less),
        Op::Eq => {
            !(cmp(lower, lit).is_some_and(|o| o == Ordering::Greater)
                || cmp(upper, lit).is_some_and(|o| o == Ordering::Less))
        },
        Op::In => p.lits.iter().any(|l| {
            !(cmp(lower, Some(l)).is_some_and(|o| o == Ordering::Greater)
                || cmp(upper, Some(l)).is_some_and(|o| o == Ordering::Less))
        }),
        Op::StartsWith => {
            let Some(Lit::Str(prefix)) = lit else {
                return true;
            };
            let prefix = prefix.as_bytes();
            let check = |bound: Option<&Lit>, reject: Ordering| match bound {
                Some(Lit::Str(b)) => {
                    let b = b.as_bytes();
                    b[..b.len().min(prefix.len())].cmp(prefix) == reject
                },
                _ => false,
            };
            !(check(lower, Ordering::Greater) || check(upper, Ordering::Less))
        },
        _ => true,
    }
}

// Column metrics.

fn metrics_might_match(filter: &Bound, file: &DataFile) -> bool {
    if file.record_count == 0 {
        return false;
    }
    if file.record_count < 0 {
        // Some format v1 writers wrote -1.
        return true;
    }
    filter.eval(&mut |p| predicate_metrics_might_match(p, file))
}

fn predicate_metrics_might_match(p: &Predicate, file: &DataFile) -> bool {
    let id = p.field_id;
    let is_float = matches!(p.ty, PrimitiveType::Float | PrimitiveType::Double);
    let value_count = file.value_count(id);
    let null_count = file.null_value_count(id);
    let nan_count = file.nan_value_count(id);

    let nulls_only = matches!((value_count, null_count), (Some(v), Some(n)) if v == n);
    let nans_only = matches!((value_count, nan_count), (Some(v), Some(n)) if v == n && n > 0);
    let may_have_nan = is_float && nan_count != Some(0);

    match p.op {
        Op::IsNull => null_count != Some(0),
        Op::NotNull => !nulls_only,
        Op::IsNan => is_float && !nulls_only && nan_count != Some(0),
        Op::NotNan => !nans_only,
        Op::NotEq | Op::NotIn | Op::NotStartsWith => !nulls_only,
        _ => {
            if nulls_only {
                return false;
            }
            if nans_only {
                return matches!(p.op, Op::Gt | Op::GtEq);
            }
            let lower = file.lower_bound(id).and_then(|b| decode_bound(b, &p.ty));
            let upper = file.upper_bound(id).and_then(|b| decode_bound(b, &p.ty));
            bounds_might_match(p, lower.as_ref(), upper.as_ref(), may_have_nan)
        },
    }
}

/// Iceberg single-value binary serialization → literal. Widths of promoted types (int → long,
/// float → double) are accepted.
pub fn decode_bound(b: &[u8], ty: &PrimitiveType) -> Option<Lit> {
    use PrimitiveType as P;
    let le_i32 = || {
        <[u8; 4]>::try_from(b)
            .ok()
            .map(|a| i64::from(i32::from_le_bytes(a)))
    };
    let le_i64 = || <[u8; 8]>::try_from(b).ok().map(i64::from_le_bytes);
    Some(match ty {
        P::Boolean => Lit::Bool(*b.first()? != 0),
        P::Int | P::Date => Lit::Int(le_i32()?),
        P::Long => Lit::Int(le_i64().or_else(le_i32)?),
        P::Time | P::Timestamp | P::Timestamptz | P::TimestampNs | P::TimestamptzNs => {
            Lit::Int(le_i64()?)
        },
        P::Float | P::Double => match b.len() {
            4 => Lit::Float(f64::from(f32::from_le_bytes(b.try_into().ok()?))),
            8 => Lit::Float(f64::from_le_bytes(b.try_into().ok()?)),
            _ => return None,
        },
        P::String => Lit::Str(std::str::from_utf8(b).ok()?.to_owned()),
        P::Binary | P::Fixed(_) | P::Uuid => Lit::Bytes(b.to_vec()),
        P::Decimal { .. } => {
            if b.len() > 16 {
                return None;
            }
            let negative = b.first().is_some_and(|x| x & 0x80 != 0);
            let mut le = if negative { [0xFF; 16] } else { [0; 16] };
            for (i, byte) in b.iter().rev().enumerate() {
                le[i] = *byte;
            }
            Lit::Decimal(i128::from_le_bytes(le))
        },
        P::Unknown => return None,
    })
}

/// A partition tuple value as a literal of the partition field's type.
fn datum_to_lit(d: &Datum, ty: &PrimitiveType) -> Option<Lit> {
    use PrimitiveType as P;
    Some(match (d, ty) {
        (Datum::Bool(b), _) => Lit::Bool(*b),
        (Datum::Int(v), _) => Lit::Int(i64::from(*v)),
        (Datum::Long(v), _) => Lit::Int(*v),
        (Datum::Float(v), _) => Lit::Float(f64::from(*v)),
        (Datum::Double(v), _) => Lit::Float(*v),
        (Datum::String(s), P::String) => Lit::Str(s.clone()),
        (Datum::Bytes(b), P::Decimal { .. }) => decode_bound(b, ty)?,
        (Datum::Bytes(b), _) => Lit::Bytes(b.clone()),
        (Datum::String(s), _) => Lit::Bytes(s.as_bytes().to_vec()),
    })
}
