use std::hash::{Hash, Hasher};
use std::sync::Arc;

use polars_core::prelude::SchemaRef;
use polars_core::scalar::Scalar;
use polars_utils::pl_str::PlSmallStr;
use serde::{Deserialize, Serialize};

use super::connection::ConnectionSpec;
use super::dialect::DialectName;

/// What a database scan reads.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum DatabaseRelation {
    /// A table, as `[catalog.][schema.]name`. Names are unquoted; `quoted` says, per part,
    /// whether the user quoted it. Bare parts are folded to the dialect's case by
    /// [`DatabaseRelation::fold_case`] once the dialect is known.
    Table {
        parts: Vec<PlSmallStr>,
        quoted: Vec<bool>,
    },
    /// A user query. Pushdown wraps it as a derived table.
    Query { sql: PlSmallStr },
}

impl DatabaseRelation {
    /// Classifies `source` as a table name or a query. A table name is one to three
    /// dot-separated parts, each a bare identifier or quoted with `"…"`, `` `…` `` or `[…]`.
    pub fn parse(source: &str, source_type: Option<&str>) -> Self {
        match source_type {
            Some("table") => {
                // A name that doesn't parse is used as written.
                let parts = split_table_name(source).unwrap_or_else(|| vec![(source.into(), true)]);
                return Self::table(parts);
            },
            Some("query") => {
                return Self::Query {
                    sql: source.trim().into(),
                };
            },
            _ => {},
        }
        match split_table_name(source) {
            Some(parts) => Self::table(parts),
            None => Self::Query {
                sql: source.trim().into(),
            },
        }
    }

    fn table(parts: Vec<(PlSmallStr, bool)>) -> Self {
        let (parts, quoted) = parts.into_iter().unzip();
        Self::Table { parts, quoted }
    }

    /// The relation with its bare (unquoted) table name parts as the database stores them:
    /// upper-cased on Oracle, Snowflake and Exasol, lower-cased on Postgres, Redshift and
    /// Trino. The others match bare names case-insensitively, or as written. Quoted parts stay
    /// as written. `None` if nothing changes.
    pub fn fold_case(&self, dialect: DialectName) -> Option<Self> {
        let Self::Table { parts, quoted } = self else {
            return None;
        };
        let fold: fn(&str) -> String = match dialect {
            DialectName::Oracle | DialectName::Snowflake | DialectName::Exasol => str::to_uppercase,
            // Postgres folds ASCII letters only (in UTF-8 databases).
            DialectName::Postgresql | DialectName::Redshift => str::to_ascii_lowercase,
            DialectName::Trino => str::to_lowercase,
            _ => return None,
        };
        let folded: Vec<PlSmallStr> = parts
            .iter()
            .zip(quoted)
            .map(|(p, &q)| if q { p.clone() } else { fold(p).into() })
            .collect();
        (folded != *parts).then(|| Self::Table {
            parts: folded,
            quoted: quoted.clone(),
        })
    }

    pub fn table_parts(&self) -> Option<&[PlSmallStr]> {
        match self {
            Self::Table { parts, .. } => Some(parts),
            Self::Query { .. } => None,
        }
    }

    pub fn table_name(&self) -> Option<&str> {
        self.table_parts()
            .and_then(|p| p.last())
            .map(|s| s.as_str())
    }

    /// The table's schema (namespace), if qualified.
    pub fn table_schema(&self) -> Option<&str> {
        let parts = self.table_parts()?;
        (parts.len() >= 2).then(|| parts[parts.len() - 2].as_str())
    }
}

/// Splits `a.b.c`, honouring quotes, into `(part, quoted)`. `None` if the string is not a table
/// name.
fn split_table_name(s: &str) -> Option<Vec<(PlSmallStr, bool)>> {
    let s = s.trim();
    if s.is_empty() {
        return None;
    }
    let mut parts = vec![];
    let mut chars = s.chars().peekable();
    loop {
        let (part, quoted) = match chars.peek()? {
            q @ ('"' | '`' | '[') => {
                let close = if *q == '[' { ']' } else { *q };
                chars.next();
                let mut part = String::new();
                loop {
                    let c = chars.next()?;
                    if c == close {
                        // A doubled closing quote is an escaped quote.
                        if chars.peek() == Some(&close) && close != ']' {
                            chars.next();
                            part.push(c);
                            continue;
                        }
                        break;
                    }
                    part.push(c);
                }
                (part, true)
            },
            c if c.is_alphabetic() || *c == '_' => {
                let mut part = String::new();
                while let Some(&c) = chars.peek() {
                    if c.is_alphanumeric() || c == '_' || c == '$' {
                        part.push(c);
                        chars.next();
                    } else {
                        break;
                    }
                }
                (part, false)
            },
            _ => return None,
        };
        parts.push((PlSmallStr::from(part), quoted));
        match chars.next() {
            None => break,
            Some('.') => continue,
            Some(_) => return None,
        }
    }
    (parts.len() <= 3).then_some(parts)
}

/// How range bounds are chosen.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PartitionStrategy {
    /// Equal-width ranges between MIN and MAX.
    #[default]
    MinMax,
}

/// The user's partitioning request. Resolved into concrete partitions during planning.
#[derive(Clone, Debug, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum PartitionSpec {
    /// A single partition.
    None,
    /// Choose per database and table (physical ranges, primary-key ranges, or none).
    Auto { num_partitions: Option<usize> },
    /// Ranges on one column.
    Column {
        column: PlSmallStr,
        num_partitions: Option<usize>,
        /// Sets the stride only. Rows outside still land in the first or last partition.
        range: Option<Box<(Scalar, Scalar)>>,
        strategy: PartitionStrategy,
    },
    /// One partition per WHERE fragment.
    Predicates(Vec<PlSmallStr>),
}

/// Consistency across partitions.
#[derive(Clone, Copy, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub enum Consistency {
    #[default]
    None,
    Snapshot,
}

#[derive(Clone, Debug, Default, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct FetchOptions {
    /// Rows per batch requested from the driver, where it supports it.
    pub batch_size: Option<usize>,
    /// Concurrent connections for the scan.
    pub max_connections: Option<usize>,
}

/// The `scan_database` request, as held in the DSL.
#[derive(Clone, Debug, Serialize, Deserialize)]
pub struct DatabaseScanOptions {
    pub connection: ConnectionSpec,
    pub relation: DatabaseRelation,
    pub dialect: Option<DialectName>,
    /// The user's driver pin (`driver=`), or `None` for auto-detection.
    pub driver: Option<PlSmallStr>,
    pub partitioning: PartitionSpec,
    pub order_by: Vec<PlSmallStr>,
    pub consistency: Consistency,
    pub fetch: FetchOptions,
    pub pushdown: bool,
    /// Applied on top of the discovered schema. A full `schema=` lives in the scan args.
    pub schema_overrides: Option<SchemaRef>,
}

impl DatabaseScanOptions {
    /// The options with the relation's bare names folded to the dialect's case
    /// ([`DatabaseRelation::fold_case`]).
    pub fn with_relation_case_folded(self: Arc<Self>, dialect: DialectName) -> Arc<Self> {
        match self.relation.fold_case(dialect) {
            Some(relation) => Arc::new(Self {
                relation,
                ..(*self).clone()
            }),
            None => self,
        }
    }

    /// The options with a query source's SQL replaced.
    pub fn with_query_sql(self: Arc<Self>, sql: PlSmallStr) -> Arc<Self> {
        Arc::new(Self {
            relation: DatabaseRelation::Query { sql },
            ..(*self).clone()
        })
    }

    /// A stable hash of the scan, for deduplication. It covers secrets too (see
    /// [`ConnectionSpec::fingerprint`]) and is never shown.
    pub fn fingerprint(&self) -> u64 {
        let mut h = polars_utils::aliases::PlFixedStateQuality::default().build_hasher();
        self.connection.fingerprint(&mut h);
        self.relation.hash(&mut h);
        self.dialect.hash(&mut h);
        self.driver.hash(&mut h);
        self.partitioning.hash(&mut h);
        self.order_by.hash(&mut h);
        self.consistency.hash(&mut h);
        self.fetch.hash(&mut h);
        self.pushdown.hash(&mut h);
        if let Some(overrides) = &self.schema_overrides {
            for (name, dtype) in overrides.iter() {
                name.hash(&mut h);
                dtype.hash(&mut h);
            }
        }
        h.finish()
    }
}

use std::hash::BuildHasher;

#[cfg(test)]
mod tests {
    use super::*;

    /// A table relation; a part is quoted if it starts with `"`.
    fn table(parts: &[&str]) -> DatabaseRelation {
        DatabaseRelation::Table {
            parts: parts
                .iter()
                .map(|s| PlSmallStr::from(s.trim_start_matches('"')))
                .collect(),
            quoted: parts.iter().map(|s| s.starts_with('"')).collect(),
        }
    }

    #[test]
    fn classify_source() {
        assert_eq!(DatabaseRelation::parse("orders", None), table(&["orders"]));
        assert_eq!(
            DatabaseRelation::parse("public.orders", None),
            table(&["public", "orders"])
        );
        assert_eq!(
            DatabaseRelation::parse(r#""My Schema"."Or""ders""#, None),
            table(&["\"My Schema", "\"Or\"ders"])
        );
        assert_eq!(
            DatabaseRelation::parse("[dbo].[t]", None),
            table(&["\"dbo", "\"t"])
        );
        assert_eq!(
            DatabaseRelation::parse("`db`.`t`", None),
            table(&["\"db", "\"t"])
        );
        assert!(matches!(
            DatabaseRelation::parse("SELECT * FROM t", None),
            DatabaseRelation::Query { .. }
        ));
        assert!(matches!(
            DatabaseRelation::parse("a.b.c.d", None),
            DatabaseRelation::Query { .. }
        ));
        assert_eq!(
            DatabaseRelation::parse("select", Some("table")),
            table(&["select"])
        );
        assert_eq!(
            DatabaseRelation::parse("not a name", Some("table")),
            table(&["\"not a name"])
        );
    }

    #[test]
    fn bare_names_fold_per_dialect() {
        let rel = DatabaseRelation::parse(r#"Sales."Mixed".Orders"#, None);
        assert_eq!(
            rel.fold_case(DialectName::Oracle),
            Some(table(&["SALES", "\"Mixed", "ORDERS"]))
        );
        assert_eq!(
            rel.fold_case(DialectName::Postgresql),
            Some(table(&["sales", "\"Mixed", "orders"]))
        );
        assert_eq!(rel.fold_case(DialectName::Mysql), None);
        assert_eq!(rel.fold_case(DialectName::Mssql), None);
        // Already in the stored case, or quoted.
        assert_eq!(
            DatabaseRelation::parse("orders", None).fold_case(DialectName::Postgresql),
            None
        );
        assert_eq!(
            DatabaseRelation::parse(r#""rv_t""#, None).fold_case(DialectName::Oracle),
            None
        );
    }
}
