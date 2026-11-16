//! Manifest and file pruning with a bound row filter ([`crate::iceberg::expr`]).
//!
//! Follows PyIceberg's planning evaluators (inclusive projection onto partition specs, the
//! manifest evaluator over partition summaries, partition value evaluation, and the inclusive
//! metrics evaluator), with one deliberate difference: NaN. Polars orders NaN above every other
//! value (`NaN > x` is true), whereas Iceberg evaluators treat comparisons with NaN as false. So
//! greater-than predicates on float columns only prune when NaNs are known to be absent.
//!
//! Null semantics are Iceberg's: `NotEq`, `NotIn` and `NotStartsWith` match null values. Polars
//! sends e.g. `~is_in(..., nulls_equal=True)` as `Not(In)`, which keeps null rows.
//!
//! Every function answers "might rows match?"; `true` keeps the manifest / file.
use std::cmp::Ordering;
use std::sync::Arc;

use polars_utils::aliases::PlHashMap;

use crate::iceberg::avro::Datum;
use crate::iceberg::expr::{Bound, Lit, Op, Predicate};
use crate::iceberg::manifest::{DataFile, FieldSummary, ManifestFile};
#[cfg(test)]
use crate::iceberg::spec::PartitionField;
use crate::iceberg::spec::{PartitionSpec, PrimitiveType, Schema, Transform, Type};
use crate::iceberg::values::parse_uuid;

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
    /// Per partition field: the maximum values of the integer types its source has had in any
    /// schema, for detecting wrapped `truncate` values (see [`may_hold_wrapped_truncate`]).
    int_maxes: Vec<Vec<i64>>,
}

impl Pruner {
    /// `schemas`: all schemas of the table, whose type history decides which wrapped
    /// `truncate` values manifests may hold.
    pub fn new(
        filter: Bound,
        specs: &PlHashMap<i32, Arc<PartitionSpec>>,
        schemas: &PlHashMap<i32, Arc<Schema>>,
    ) -> Self {
        let projections = specs
            .iter()
            .map(|(id, spec)| {
                (
                    *id,
                    SpecProjection {
                        filter: project(&filter, spec),
                        int_maxes: spec
                            .fields
                            .iter()
                            .map(|f| source_int_maxes(f.source_id, schemas))
                            .collect(),
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
            let idx = p.field_id as usize;
            let decode = |b: &Option<Vec<u8>>| b.as_deref().and_then(|b| decode_bound(b, &ty));
            let lower = decode(&summary.lower_bound);
            let upper = decode(&summary.upper_bound);
            // Wrapped values sort above every truncated value, so they can lie strictly between
            // the bounds, also when the upper bound lies beyond their range (e.g. a merged
            // manifest of files from before and after an int -> long promotion).
            let wrapped = [&lower, &upper]
                .into_iter()
                .flatten()
                .any(|v| is_wrapped_truncate(spec, idx, v))
                || may_hold_wrapped_truncate(
                    spec,
                    idx,
                    &projection.int_maxes[idx],
                    lower.as_ref(),
                    upper.as_ref(),
                );
            wrapped || summary_might_match(p, summary, &ty)
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
                                Some(v) if is_wrapped_truncate(spec, idx, &v) => return true,
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

/// Whether a partition value of a `truncate` field is not a truncated value. Writers (Java,
/// PyIceberg) truncate integers within the width of the type's minimum with a wrapping
/// subtraction, giving a large positive value. Adding 2^32 or 2^64 to a multiple of the width
/// never gives one when the subtraction wraps, so these values are not multiples of the width.
fn is_wrapped_truncate(spec: &PartitionSpec, idx: usize, value: &Lit) -> bool {
    match (spec.fields.get(idx).map(|f| &f.transform), value) {
        (Some(Transform::Truncate(width)), Lit::Int(v)) if *width > 0 => {
            v.rem_euclid(i64::from(*width)) != 0
        },
        _ => false,
    }
}

/// Whether the partition values of a `truncate` field between `lower` and `upper` (unbounded if
/// unknown) can hold a wrapped value: a wrapped value `v - (v mod w) + 2^n` (for `v` within `w`
/// of the minimum) is in `MAX - w + 2 ..= MAX`. Every integer type the source has had counts:
/// after an int → long promotion, manifests written before it hold values wrapped within the
/// int range.
fn may_hold_wrapped_truncate(
    spec: &PartitionSpec,
    idx: usize,
    int_maxes: &[i64],
    lower: Option<&Lit>,
    upper: Option<&Lit>,
) -> bool {
    let Some(Transform::Truncate(width)) = spec.fields.get(idx).map(|f| &f.transform) else {
        return false;
    };
    if *width == 0 {
        return false;
    }
    let as_int = |v: Option<&Lit>, unbounded: i64| match v {
        None => Some(unbounded),
        Some(Lit::Int(v)) => Some(*v),
        Some(_) => None,
    };
    let (Some(lower), Some(upper)) = (as_int(lower, i64::MIN), as_int(upper, i64::MAX)) else {
        return false;
    };
    int_maxes
        .iter()
        .any(|max| lower <= *max && upper > max - i64::from(*width) + 1)
}

/// Maximum values of the integer types a field may have had, judged from `schemas`. A long field
/// counts as having been an int too: metadata cleanup can remove unused schemas, so the
/// remaining ones do not prove that an int -> long promotion never happened.
fn source_int_maxes(source_id: i32, schemas: &PlHashMap<i32, Arc<Schema>>) -> Vec<i64> {
    let mut maxes = vec![];
    for schema in schemas.values() {
        let field_maxes: &[i64] = match schema.field_by_id(source_id).map(|f| &f.field_type) {
            Some(Type::Primitive(PrimitiveType::Int)) => &[i32::MAX as i64],
            Some(Type::Primitive(PrimitiveType::Long)) => &[i32::MAX as i64, i64::MAX],
            _ => continue,
        };
        for max in field_maxes {
            if !maxes.contains(max) {
                maxes.push(*max);
            }
        }
    }
    maxes
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
            let Some(lits) = apply_all(&t) else {
                return Bound::True;
            };
            // Older Java writers rounded pre-epoch values toward zero instead of down, so a
            // negative partition value may be one above `t(v)`. As in Java's
            // `ProjectionUtil.fixInclusiveTimeProjection`, upper bounds and equalities on
            // negative values also accept `t(v) + 1`.
            let fix_negative = match transform {
                Transform::Year | Transform::Month => true,
                _ => !matches!(p.ty, PrimitiveType::Date),
            };
            let bumped = |v: &Lit| match v {
                Lit::Int(v) if fix_negative && *v < 0 => Some(Lit::Int(v + 1)),
                _ => None,
            };
            // Transforms are monotonic (floor), so `x < v` implies `t(x) <= t(v)`.
            match p.op {
                Op::Lt | Op::LtEq => {
                    let lits = lits.iter().map(|l| bumped(l).unwrap_or_else(|| l.clone()));
                    with(Op::LtEq, lits.collect(), PrimitiveType::Int)
                },
                Op::Gt | Op::GtEq => with(Op::GtEq, lits, PrimitiveType::Int),
                Op::Eq | Op::In => {
                    let extra: Vec<Lit> = lits.iter().filter_map(bumped).collect();
                    let op = if p.op == Op::Eq && extra.is_empty() {
                        Op::Eq
                    } else {
                        Op::In
                    };
                    with(
                        op,
                        lits.into_iter().chain(extra).collect(),
                        PrimitiveType::Int,
                    )
                },
                _ => Bound::True,
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
        Transform::Bucket(n) => {
            let b = |l: &Lit| bucket(*n, &p.ty, l).map(Lit::Int);
            let (op, lits) = match p.op {
                Op::Eq => (Op::Eq, apply_all(&b)),
                Op::In => (
                    Op::In,
                    apply_all(&b).map(|mut lits| {
                        lits.sort_by(|a, b| a.partial_cmp(b).unwrap_or(Ordering::Equal));
                        lits.dedup();
                        lits
                    }),
                ),
                _ => return Bound::True,
            };
            match lits {
                Some(lits) => with(op, lits, PrimitiveType::Int),
                None => Bound::True,
            }
        },
        Transform::Void | Transform::Other(_) => Bound::True,
    }
}

/// The bucket of a value, as defined by the Iceberg spec (32-bit Murmur3 hash, seed 0, of the
/// value's bucket serialization); `None` for types this does not handle.
fn bucket(n: u32, ty: &PrimitiveType, lit: &Lit) -> Option<i64> {
    use PrimitiveType as P;
    if n == 0 {
        return None;
    }
    let hash = match (ty, lit) {
        // Ints are hashed as longs, so a column promoted from int to long keeps its buckets.
        (P::Int | P::Long | P::Date | P::Time | P::Timestamp | P::Timestamptz, Lit::Int(v)) => {
            murmur3_32(&v.to_le_bytes())
        },
        (P::Decimal { .. }, Lit::Decimal(v)) => murmur3_32(minimal_be_bytes(*v).as_slice()),
        (P::String, Lit::Str(s)) => murmur3_32(s.as_bytes()),
        (P::Uuid | P::Binary | P::Fixed(_), Lit::Bytes(b)) => murmur3_32(b),
        _ => return None,
    };
    Some(i64::from((hash & i32::MAX as u32) % n))
}

/// Minimal big-endian two's-complement bytes of an unscaled decimal (Java's
/// `BigInteger.toByteArray`).
fn minimal_be_bytes(v: i128) -> Vec<u8> {
    let be = v.to_be_bytes();
    let redundant = be
        .windows(2)
        .take_while(|w| (w[0] == 0x00 && w[1] & 0x80 == 0) || (w[0] == 0xFF && w[1] & 0x80 != 0))
        .count();
    be[redundant..].to_vec()
}

/// MurmurHash3 x86 32-bit, seed 0.
fn murmur3_32(data: &[u8]) -> u32 {
    const C1: u32 = 0xcc9e_2d51;
    const C2: u32 = 0x1b87_3593;
    let mix = |k: u32| k.wrapping_mul(C1).rotate_left(15).wrapping_mul(C2);

    let mut h: u32 = 0;
    let mut chunks = data.chunks_exact(4);
    for chunk in &mut chunks {
        h ^= mix(u32::from_le_bytes(chunk.try_into().unwrap()));
        h = h.rotate_left(13).wrapping_mul(5).wrapping_add(0xe654_6b64);
    }
    let tail = chunks.remainder();
    if !tail.is_empty() {
        let mut k: u32 = 0;
        for (i, b) in tail.iter().enumerate() {
            k |= u32::from(*b) << (8 * i);
        }
        h ^= mix(k);
    }

    h ^= data.len() as u32;
    h ^= h >> 16;
    h = h.wrapping_mul(0x85eb_ca6b);
    h ^= h >> 13;
    h = h.wrapping_mul(0xc2b2_ae35);
    h ^ (h >> 16)
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
        // Values within `width` of `i64::MIN` have no truncated value: not projected.
        Lit::Int(v) if w > 0 => v.checked_sub(v.rem_euclid(w)).map(Lit::Int),
        Lit::Str(s) => Some(Lit::Str(s.chars().take(width as usize).collect())),
        Lit::Bytes(b) => Some(Lit::Bytes(b.iter().take(width as usize).copied().collect())),
        _ => None,
    }
}

// Evaluation of an exact value (partition tuples).

fn value_might_match(p: &Predicate, value: Option<&Lit>) -> bool {
    let Some(value) = value else {
        // Comparisons with null never hold, but their negations do.
        return matches!(p.op, Op::IsNull | Op::NotEq | Op::NotIn | Op::NotStartsWith);
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

    // Absent bounds are unknown: the spec makes them optional, so they do not imply that all
    // values are null (as Java and PyIceberg assume).
    match p.op {
        Op::IsNull => s.contains_null,
        Op::IsNan => may_have_nan,
        Op::NotNull | Op::NotNan | Op::NotEq | Op::NotIn | Op::NotStartsWith => true,
        _ => bounds_might_match(p, lower.as_ref(), upper.as_ref(), may_have_nan),
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
        // These match nulls.
        Op::NotEq | Op::NotIn | Op::NotStartsWith => true,
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
        // A UUID in an Avro `string` is in its text form.
        (Datum::String(s), P::Uuid) => Lit::Bytes(parse_uuid(s)?),
        (Datum::String(s), _) => Lit::Bytes(s.as_bytes().to_vec()),
    })
}

#[cfg(test)]
mod tests {
    use std::sync::Arc;

    use super::*;

    fn project_one(op: Op, ty: PrimitiveType, lits: Vec<Lit>, transform: Transform) -> Bound {
        let p = Predicate {
            field_id: 1,
            ty,
            op,
            lits,
        };
        project_predicate(&p, 0, &transform)
    }

    fn pred(bound: &Bound) -> (Op, Vec<Lit>) {
        match bound {
            Bound::Pred(p) => (p.op, p.lits.clone()),
            other => panic!("expected a predicate, got {other:?}"),
        }
    }

    #[test]
    fn negated_predicates_match_null_partition_values() {
        let p = |op, lits| Predicate {
            field_id: 1,
            ty: PrimitiveType::Long,
            op,
            lits,
        };
        let one = || vec![Lit::Int(1)];
        // E.g. `~is_in([1], nulls_equal=True)` keeps nulls.
        assert!(value_might_match(&p(Op::NotIn, one()), None));
        assert!(value_might_match(&p(Op::NotEq, one()), None));
        assert!(value_might_match(&p(Op::IsNull, vec![]), None));
        assert!(!value_might_match(&p(Op::In, one()), None));
        assert!(!value_might_match(&p(Op::Eq, one()), None));
        assert!(!value_might_match(&p(Op::Lt, one()), None));
        assert!(!value_might_match(&p(Op::NotNull, vec![]), None));
    }

    #[test]
    fn absent_summary_bounds_are_unknown() {
        let p = |op, lits| Predicate {
            field_id: 1,
            ty: PrimitiveType::Long,
            op,
            lits,
        };
        let summary = |contains_null| FieldSummary {
            contains_null,
            contains_nan: None,
            lower_bound: None,
            upper_bound: None,
        };
        let ty = PrimitiveType::Long;
        for contains_null in [false, true] {
            let s = summary(contains_null);
            assert!(summary_might_match(&p(Op::Eq, vec![Lit::Int(2)]), &s, &ty));
            assert!(summary_might_match(&p(Op::Lt, vec![Lit::Int(2)]), &s, &ty));
            assert!(summary_might_match(&p(Op::In, vec![Lit::Int(2)]), &s, &ty));
            assert!(summary_might_match(&p(Op::NotNull, vec![]), &s, &ty));
            assert_eq!(
                summary_might_match(&p(Op::IsNull, vec![]), &s, &ty),
                contains_null
            );
        }
    }

    // 1969-12-31T12:00:00 in microseconds: day -1.
    const PRE_EPOCH_US: i64 = -12 * 3_600_000_000;

    #[test]
    fn negative_time_projection_accepts_values_rounded_toward_zero() {
        let ts = PrimitiveType::Timestamp;
        let lit = || vec![Lit::Int(PRE_EPOCH_US)];

        let bound = project_one(Op::Lt, ts.clone(), lit(), Transform::Day);
        assert_eq!(pred(&bound), (Op::LtEq, vec![Lit::Int(0)]));

        let bound = project_one(Op::Eq, ts.clone(), lit(), Transform::Day);
        assert_eq!(pred(&bound), (Op::In, vec![Lit::Int(-1), Lit::Int(0)]));

        let bound = project_one(Op::In, ts.clone(), lit(), Transform::Hour);
        assert_eq!(pred(&bound), (Op::In, vec![Lit::Int(-12), Lit::Int(-11)]));

        // Lower bounds are unaffected: rounding toward zero only increases negative values.
        let bound = project_one(Op::GtEq, ts.clone(), lit(), Transform::Day);
        assert_eq!(pred(&bound), (Op::GtEq, vec![Lit::Int(-1)]));

        // Non-negative values are exact.
        let bound = project_one(Op::Eq, ts, vec![Lit::Int(0)], Transform::Day);
        assert_eq!(pred(&bound), (Op::Eq, vec![Lit::Int(0)]));
    }

    #[test]
    fn bucket_hashes_match_spec() {
        // Test vectors from the Iceberg spec, "Appendix B: 32-bit Hash Requirements".
        let hash = |ty: PrimitiveType, lit: Lit| -> i32 {
            match (&ty, &lit) {
                (_, Lit::Int(v)) => murmur3_32(&v.to_le_bytes()) as i32,
                (_, Lit::Decimal(v)) => murmur3_32(&minimal_be_bytes(*v)) as i32,
                (_, Lit::Str(s)) => murmur3_32(s.as_bytes()) as i32,
                (_, Lit::Bytes(b)) => murmur3_32(b) as i32,
                _ => unreachable!(),
            }
        };
        assert_eq!(hash(PrimitiveType::Int, Lit::Int(34)), 2017239379);
        assert_eq!(hash(PrimitiveType::Date, Lit::Int(17486)), -653330422);
        assert_eq!(
            hash(PrimitiveType::Time, Lit::Int(81_068_000_000)),
            -662762989
        );
        assert_eq!(
            hash(PrimitiveType::Timestamp, Lit::Int(1_510_871_468_000_000)),
            -2047944441
        );
        let dec = PrimitiveType::Decimal {
            precision: 9,
            scale: 2,
        };
        assert_eq!(hash(dec, Lit::Decimal(1420)), -500754589);
        assert_eq!(
            hash(PrimitiveType::String, Lit::Str("iceberg".into())),
            1210000089
        );
        let uuid = hex::decode("f79c3e09677c4bbda4793f349cb785e7").unwrap();
        assert_eq!(hash(PrimitiveType::Uuid, Lit::Bytes(uuid)), 1488055340);
        assert_eq!(
            hash(PrimitiveType::Binary, Lit::Bytes(vec![0, 1, 2, 3])),
            -188683207
        );

        // Negative decimals use the minimal two's complement.
        assert_eq!(minimal_be_bytes(-1), vec![0xFF]);
        assert_eq!(minimal_be_bytes(128), vec![0x00, 0x80]);
        assert_eq!(minimal_be_bytes(-129), vec![0xFF, 0x7F]);
    }

    #[test]
    fn bucket_projection() {
        let bound = project_one(
            Op::Eq,
            PrimitiveType::Long,
            vec![Lit::Int(34)],
            Transform::Bucket(16),
        );
        assert_eq!(
            pred(&bound),
            (Op::Eq, vec![Lit::Int(i64::from(2017239379 % 16))])
        );

        let bound = project_one(
            Op::In,
            PrimitiveType::Long,
            vec![Lit::Int(34), Lit::Int(34)],
            Transform::Bucket(16),
        );
        assert_eq!(pred(&bound).1.len(), 1);

        // Range predicates cannot be projected onto buckets.
        let bound = project_one(
            Op::Lt,
            PrimitiveType::Long,
            vec![Lit::Int(34)],
            Transform::Bucket(16),
        );
        assert!(matches!(bound, Bound::True));
    }

    #[test]
    fn truncate_projection_does_not_overflow() {
        let bound = project_one(
            Op::GtEq,
            PrimitiveType::Long,
            vec![Lit::Int(i64::MIN)],
            Transform::Truncate(10),
        );
        assert!(matches!(bound, Bound::True));

        let bound = project_one(
            Op::In,
            PrimitiveType::Long,
            vec![Lit::Int(5), Lit::Int(i64::MIN + 1)],
            Transform::Truncate(10),
        );
        assert!(matches!(bound, Bound::True));

        // Values that fit are rounded toward negative infinity.
        let bound = project_one(
            Op::GtEq,
            PrimitiveType::Long,
            vec![Lit::Int(i64::MIN + 8)],
            Transform::Truncate(10),
        );
        assert_eq!(pred(&bound), (Op::GtEq, vec![Lit::Int(i64::MIN + 8)]));
        let bound = project_one(
            Op::Eq,
            PrimitiveType::Long,
            vec![Lit::Int(-1)],
            Transform::Truncate(10),
        );
        assert_eq!(pred(&bound), (Op::Eq, vec![Lit::Int(-10)]));
    }

    #[test]
    fn wrapped_truncate_partition_values() {
        let spec = |transform| PartitionSpec {
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                transform,
            }],
        };
        let truncate = spec(Transform::Truncate(10));
        // `truncate[10]` of `i64::MIN` / `i32::MIN` as written by Java and PyIceberg.
        for v in [
            i64::MIN.wrapping_sub(i64::MIN.rem_euclid(10)),
            i64::from(i32::MIN.wrapping_sub(i32::MIN.rem_euclid(10))),
        ] {
            assert!(v > 0);
            assert!(is_wrapped_truncate(&truncate, 0, &Lit::Int(v)));
        }
        assert!(!is_wrapped_truncate(&truncate, 0, &Lit::Int(-10)));
        assert!(!is_wrapped_truncate(&truncate, 0, &Lit::Int(0)));
        assert!(!is_wrapped_truncate(
            &spec(Transform::Identity),
            0,
            &Lit::Int(7)
        ));
    }

    /// A table with `x` partitioned by `truncate[100]`, with `x` of the given type in each schema
    /// (the last one is scanned), and a manifest with partition summary bounds.
    fn wrapped_truncate_pruner(
        types: &[PrimitiveType],
        op: Op,
        lit: i64,
    ) -> (Pruner, Arc<PartitionSpec>, Arc<Schema>) {
        use crate::iceberg::spec::NestedField;

        let spec = Arc::new(PartitionSpec {
            fields: vec![PartitionField {
                source_id: 1,
                field_id: 1000,
                transform: Transform::Truncate(100),
            }],
        });
        let schemas: PlHashMap<i32, Arc<Schema>> = types
            .iter()
            .enumerate()
            .map(|(id, ty)| {
                let field = NestedField {
                    id: 1,
                    name: "x".into(),
                    required: false,
                    field_type: Type::Primitive(ty.clone()),
                    initial_default: None,
                };
                (id as i32, Arc::new(Schema::new(id as i32, vec![field])))
            })
            .collect();
        let filter = Bound::Pred(Predicate {
            field_id: 1,
            ty: types.last().unwrap().clone(),
            op,
            lits: vec![Lit::Int(lit)],
        });
        let pruner = Pruner::new(filter, &PlHashMap::from_iter([(0, spec.clone())]), &schemas);
        let scanned = schemas[&(types.len() as i32 - 1)].clone();
        (pruner, spec, scanned)
    }

    fn summary_manifest(lower: Vec<u8>, upper: Vec<u8>) -> ManifestFile {
        ManifestFile {
            path: String::new(),
            length: 0,
            spec_id: 0,
            content: 0,
            sequence_number: 0,
            min_sequence_number: 0,
            added_snapshot_id: None,
            partitions: vec![FieldSummary {
                contains_null: false,
                contains_nan: None,
                lower_bound: Some(lower),
                upper_bound: Some(upper),
            }],
        }
    }

    #[test]
    fn manifest_with_wrapped_truncate_value_between_bounds_is_kept() {
        let (pruner, spec, schema) = wrapped_truncate_pruner(&[PrimitiveType::Int], Op::Lt, 0);
        let manifest = |lower: i32, upper: i32| {
            summary_manifest(lower.to_le_bytes().to_vec(), upper.to_le_bytes().to_vec())
        };

        // `truncate[100]` of `i32::MIN` wraps to 2147483596, between these bounds.
        assert!(pruner.manifest_might_match(&manifest(2147483400, 2147483600), &spec, &schema));
        assert!(!pruner.manifest_might_match(&manifest(100, 2147483500), &spec, &schema));
    }

    #[test]
    fn wrapped_int_truncate_value_is_kept_after_promotion_to_long() {
        use PrimitiveType::{Int, Long};

        // `x <= -5` on a column promoted from int to long. Manifests written before the
        // promotion hold `truncate[100]` of `i32::MIN` wrapped to 2147483596.
        let (pruner, spec, schema) = wrapped_truncate_pruner(&[Int, Long], Op::LtEq, -5);
        let int_bounds = summary_manifest(
            100_i32.to_le_bytes().to_vec(),
            2147483600_i32.to_le_bytes().to_vec(),
        );
        assert!(pruner.manifest_might_match(&int_bounds, &spec, &schema));
        // The same values in a manifest rewritten after the promotion.
        let long_bounds = summary_manifest(
            100_i64.to_le_bytes().to_vec(),
            2147483600_i64.to_le_bytes().to_vec(),
        );
        assert!(pruner.manifest_might_match(&long_bounds, &spec, &schema));
        assert!(!pruner.manifest_might_match(
            &summary_manifest(
                100_i64.to_le_bytes().to_vec(),
                2147483500_i64.to_le_bytes().to_vec(),
            ),
            &spec,
            &schema
        ));

        // A manifest merged from files of before and after the promotion: its upper bound lies
        // beyond the int range, with the wrapped value between the bounds.
        let merged_bounds = summary_manifest(
            2147483400_i64.to_le_bytes().to_vec(),
            3000000000_i64.to_le_bytes().to_vec(),
        );
        assert!(pruner.manifest_might_match(&merged_bounds, &spec, &schema));
        assert!(!pruner.manifest_might_match(
            &summary_manifest(
                2147483700_i64.to_le_bytes().to_vec(),
                3000000000_i64.to_le_bytes().to_vec(),
            ),
            &spec,
            &schema
        ));

        // The int schema may have been removed by metadata cleanup, after which only the long
        // schema remains. The manifests still hold the wrapped int value.
        let (pruner, spec, schema) = wrapped_truncate_pruner(&[Long], Op::LtEq, -5);
        assert!(pruner.manifest_might_match(&int_bounds, &spec, &schema));
        assert!(pruner.manifest_might_match(&long_bounds, &spec, &schema));
        assert!(pruner.manifest_might_match(&merged_bounds, &spec, &schema));
        assert!(!pruner.manifest_might_match(
            &summary_manifest(
                100_i64.to_le_bytes().to_vec(),
                2147483500_i64.to_le_bytes().to_vec(),
            ),
            &spec,
            &schema
        ));
    }

    #[test]
    fn string_encoded_uuid_partition_value_matches_literal() {
        let bytes: Vec<u8> = (0..16).collect();
        let text = Datum::String("00010203-0405-0607-0809-0a0b0c0d0e0f".into());
        assert_eq!(
            datum_to_lit(&text, &PrimitiveType::Uuid),
            Some(Lit::Bytes(bytes))
        );
        // Unparseable values are unknown (the file is kept), not compared as text.
        let invalid = Datum::String("not-a-uuid".into());
        assert_eq!(datum_to_lit(&invalid, &PrimitiveType::Uuid), None);
    }

    #[test]
    fn negative_date_projection() {
        // 1969-12-31: month -1.
        let bound = project_one(
            Op::Eq,
            PrimitiveType::Date,
            vec![Lit::Int(-1)],
            Transform::Month,
        );
        assert_eq!(pred(&bound), (Op::In, vec![Lit::Int(-1), Lit::Int(0)]));

        // `day` of a date is the date itself: no adjustment.
        let bound = project_one(
            Op::Eq,
            PrimitiveType::Date,
            vec![Lit::Int(-1)],
            Transform::Day,
        );
        assert_eq!(pred(&bound), (Op::Eq, vec![Lit::Int(-1)]));
    }
}
