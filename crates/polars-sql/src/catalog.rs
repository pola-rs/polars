//! Catalogs attached to a [`SQLContext`](crate::SQLContext): resolving table names to tables
//! of a catalog.
use std::ops::ControlFlow;

use polars_core::prelude::*;
use polars_utils::aliases::PlHashMap;
use sqlparser::ast::{
    BinaryOperator, ColumnDef, ColumnOption, CreateTableOptions, DataType as SQLDataType, Expr,
    Function, FunctionArg, FunctionArgExpr, FunctionArguments, HiveDistributionStyle, Ident,
    ObjectName, ObjectNamePart, Query, SetExpr, SqlOption, Statement, TableFactor, UnaryOperator,
    Value, ValueWithSpan, Visitor, VisitorMut, With,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::Parser;
use sqlparser::tokenizer::{Location, Span};

use crate::dyn_visit;
use crate::sql_visitors::expr_references_any_column;
use crate::time_travel::{VersionSpec, is_version_part, split_version, with_version};
use crate::types::map_sql_dtype_to_polars;

/// A table of an attached catalog that a statement reads.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CatalogTable {
    /// The name the catalog was attached under.
    pub catalog: String,
    /// The table's identifier in the catalog: namespace parts, then the table name.
    pub identifier: Vec<String>,
}

/// A catalog table that a query reads, at a version or as it is now.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogRead {
    /// The table read.
    pub table: CatalogTable,
    /// The version read; `None` for the current snapshot.
    pub version: Option<VersionSpec>,
    /// The name to pass the scan of this read back under (see
    /// `SQLContext::execute_with_catalog_tables`): the full name, then the version if any.
    pub key: Vec<String>,
}

/// A namespace of an attached catalog.
#[derive(Clone, Debug, PartialEq, Eq, Hash)]
pub struct CatalogNamespace {
    /// The name the catalog was attached under.
    pub catalog: String,
    /// The namespace parts.
    pub namespace: Vec<String>,
}

/// A DDL or DML statement on a catalog table or schema, for the caller to perform (see
/// [`SQLContext::catalog_statement`](crate::SQLContext::catalog_statement)).
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogStatement {
    /// `DROP TABLE [IF EXISTS] <name> [PURGE]`.
    DropTable {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table it resolves to.
        table: CatalogTable,
        /// `IF EXISTS`.
        if_exists: bool,
        /// `PURGE`: also delete the data files.
        purge: bool,
    },
    /// `CREATE TABLE [IF NOT EXISTS] <name> (<column>, ...)`, or `... <name> AS <query>`.
    CreateTable {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to create.
        table: CatalogTable,
        /// `IF NOT EXISTS`.
        if_not_exists: bool,
        /// The columns; empty with a query.
        schema: Schema,
        /// The query that fills the table, as SQL.
        query: Option<String>,
        /// The columns declared `NOT NULL`.
        required: Vec<String>,
        /// The table location, from `LOCATION` or the `location` option.
        location: Option<String>,
        /// The other `WITH` options, as table properties.
        properties: Vec<(String, String)>,
        /// The partition fields, in the order written; empty for an unpartitioned table.
        partition_by: Vec<CatalogPartitionField>,
    },
    /// `CREATE SCHEMA [IF NOT EXISTS] <name> [WITH (...)]`.
    CreateSchema {
        /// The schema name as the statement wrote it.
        name: String,
        /// The catalog namespace to create.
        namespace: CatalogNamespace,
        /// `IF NOT EXISTS`.
        if_not_exists: bool,
        /// The `WITH` options, as namespace properties.
        properties: Vec<(String, String)>,
    },
    /// `DROP SCHEMA [IF EXISTS] <name> [RESTRICT]`.
    DropSchema {
        /// The schema name as the statement wrote it.
        name: String,
        /// The catalog namespace to drop.
        namespace: CatalogNamespace,
        /// `IF EXISTS`.
        if_exists: bool,
    },
    /// `INSERT INTO <name> [(<column>, ...)] <query>`, or `INSERT OVERWRITE ...`.
    Insert {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to write.
        table: CatalogTable,
        /// The columns written, as listed; empty for all, in the table's order.
        columns: Vec<String>,
        /// The query giving the rows, as SQL.
        query: String,
        /// `OVERWRITE`: replace the table's rows.
        overwrite: bool,
    },
    /// `DELETE FROM <name> [WHERE <condition>]`.
    Delete {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to delete rows from.
        table: CatalogTable,
        /// The `WHERE` condition, as SQL, for messages.
        condition: Option<String>,
        /// Whether the condition has a subquery, which a catalog filter can't express.
        subquery: bool,
        /// The first part of the condition a catalog filter likely can't express, for messages.
        hint_unsupported_part: Option<String>,
    },
    /// `UPDATE <name> SET <column> = <value>, ... [WHERE <condition>]`.
    Update {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to update.
        table: CatalogTable,
        /// The `SET` columns, in the order written.
        columns: Vec<String>,
        /// The `WHERE` condition, as SQL, for messages.
        condition: Option<String>,
        /// Whether the condition has a subquery, which a catalog filter can't express.
        subquery: bool,
        /// The first part of the condition a catalog filter likely can't express, for messages.
        hint_unsupported_part: Option<String>,
    },
    /// `MERGE INTO <name> USING <source> ON <condition> WHEN ...`.
    Merge {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to merge into.
        table: CatalogTable,
    },
    /// `TRUNCATE [TABLE] [IF EXISTS] <name>`.
    Truncate {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to empty.
        table: CatalogTable,
        /// `IF EXISTS`.
        if_exists: bool,
    },
    /// `ALTER TABLE [IF EXISTS] <name> <operation>`.
    AlterTable {
        /// The table name as the statement wrote it.
        name: String,
        /// The catalog table to change.
        table: CatalogTable,
        /// `IF EXISTS`.
        if_exists: bool,
        /// The change, one per statement.
        operation: CatalogAlter,
    },
}

/// The source of a catalog `MERGE`, planned by [`crate::SQLContext::catalog_merge_source`].
pub struct MergeSourcePlan {
    /// The `USING` relation, with the columns the statement reads.
    pub source: polars_lazy::frame::LazyFrame,
    /// Per key of `ON` (`target_column = source_expression`), the target column and the key
    /// over the source's columns.
    pub keys: Vec<(String, polars_plan::dsl::Expr)>,
}

/// A planned `WHEN` clause of a catalog `MERGE`.
pub struct MergeClausePlan {
    /// `update`, `delete`, `nothing` or `insert`.
    pub kind: &'static str,
    /// The clause's number in the statement, from 1.
    pub number: usize,
    /// Per target column written, its value over the columns of the plan's frame.
    pub columns: Vec<(String, polars_plan::dsl::Expr)>,
}

/// A catalog `MERGE`, planned by [`crate::SQLContext::catalog_merge_plan`].
pub struct MergePlan {
    /// The pairs of a target and a source row matching `ON`, with the index of the matched
    /// clause that applies (null if none does).
    pub pairs: polars_lazy::frame::LazyFrame,
    /// The source rows, with the index of the not-matched clause that applies (null if none).
    pub inserts: polars_lazy::frame::LazyFrame,
    /// The `WHEN MATCHED` clauses, in order.
    pub matched: Vec<MergeClausePlan>,
    /// The `WHEN NOT MATCHED` clauses, in order.
    pub not_matched: Vec<MergeClausePlan>,
}

/// A change to a catalog table's top-level columns, properties or partitioning, by an `ALTER TABLE`.
#[derive(Clone, Debug, PartialEq, Eq)]
pub enum CatalogAlter {
    /// `ADD [COLUMN] [IF NOT EXISTS] <name> <type>`: a nullable column.
    AddColumn {
        /// The column name, as written.
        name: String,
        /// The column type.
        dtype: DataType,
        /// `IF NOT EXISTS`.
        if_not_exists: bool,
    },
    /// `DROP [COLUMN] [IF EXISTS] <name>`.
    DropColumn {
        /// The column name, as written.
        name: String,
        /// `IF EXISTS`.
        if_exists: bool,
    },
    /// `RENAME COLUMN <old> TO <new>`.
    RenameColumn {
        /// The current column name.
        old: String,
        /// The new column name.
        new: String,
    },
    /// `ALTER COLUMN <name> [SET DATA] TYPE <type>`.
    SetDataType {
        /// The column name, as written.
        name: String,
        /// The new column type.
        dtype: DataType,
    },
    /// `ALTER COLUMN <name> DROP NOT NULL`.
    DropNotNull {
        /// The column name, as written.
        name: String,
    },
    /// `SET TBLPROPERTIES ('key' = value, ...)`.
    SetProperties {
        /// The properties, in the order written.
        properties: Vec<(String, String)>,
    },
    /// `SET PARTITIONED BY (...)`, or `RESET PARTITIONED BY` (no fields): the table's partition
    /// spec from now on.
    SetPartitioning {
        /// The partition fields, in the order written.
        fields: Vec<CatalogPartitionField>,
    },
}

/// A partition field of a catalog table: an Iceberg transform of a column or a struct field.
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct CatalogPartitionField {
    /// The source as written: a column, then the struct fields down to the source (`s.f`).
    pub path: Vec<String>,
    /// The transform in Iceberg's notation: `identity`, `year`, `month`, `day`, `hour`,
    /// `bucket[N]` or `truncate[W]`.
    pub transform: String,
}

impl CatalogTable {
    /// The catalog name followed by the identifier, the name the table is known by in SQL.
    pub fn full_name(&self) -> Vec<String> {
        std::iter::once(self.catalog.clone())
            .chain(self.identifier.iter().cloned())
            .collect()
    }
}

impl CatalogNamespace {
    /// The catalog name followed by the namespace parts.
    pub fn full_name(&self) -> Vec<String> {
        std::iter::once(self.catalog.clone())
            .chain(self.namespace.iter().cloned())
            .collect()
    }
}

#[derive(Clone, Debug)]
struct AttachedCatalog {
    name: String,
    /// The namespace of one-part table names.
    default_namespace: Option<Vec<String>>,
}

/// The catalogs attached to a context.
#[derive(Clone, Debug, Default)]
pub(crate) struct CatalogRegistry {
    catalogs: Vec<AttachedCatalog>,
    /// The catalog of table names that don't start with an attached catalog.
    default: Option<String>,
}

impl CatalogRegistry {
    pub(crate) fn is_empty(&self) -> bool {
        self.catalogs.is_empty()
    }

    /// Attach a catalog; the same name replaces it, with the new `default_namespace`, and keeps
    /// its default status (`default` only ever sets the default). A name that differs from an
    /// attached one only in case is an error, so matching names ignoring case is never ambiguous.
    pub(crate) fn register(
        &mut self,
        name: &str,
        default_namespace: Option<Vec<String>>,
        default: bool,
    ) -> PolarsResult<()> {
        polars_ensure!(!name.is_empty(), InvalidOperation: "catalog name must not be empty");
        polars_ensure!(
            default_namespace.as_ref().is_none_or(|ns| !ns.is_empty() && ns.iter().all(|p| !p.is_empty())),
            InvalidOperation: "the default namespace of catalog '{}' must have non-empty parts", name
        );
        if let Some(other) = self
            .catalogs
            .iter()
            .find(|c| c.name != name && c.name.eq_ignore_ascii_case(name))
        {
            polars_bail!(
                InvalidOperation: "catalog '{}' differs from attached catalog '{}' only in case; use that name, or unregister it first",
                name, other.name
            );
        }
        let keep_default = self.default.as_deref() == Some(name);
        self.unregister(name);
        self.catalogs.push(AttachedCatalog {
            name: name.to_string(),
            default_namespace,
        });
        if default || keep_default {
            self.default = Some(name.to_string());
        }
        Ok(())
    }

    /// The name of the default catalog.
    pub(crate) fn default_catalog(&self) -> Option<&str> {
        self.default.as_deref()
    }

    /// Make an attached catalog (exact name, else ignoring case) the default.
    pub(crate) fn set_default(&mut self, name: &str) -> PolarsResult<()> {
        let Some(name) = self.catalog_name(name).map(str::to_string) else {
            polars_bail!(SQLInterface: "catalog '{}' is not attached", name);
        };
        self.default = Some(name);
        Ok(())
    }

    /// Detach a catalog (exact name, else ignoring case), returning the name it had.
    pub(crate) fn unregister(&mut self, name: &str) -> Option<String> {
        let idx = self.position(name)?;
        let removed = self.catalogs.remove(idx).name;
        if self.default.as_deref() == Some(removed.as_str()) {
            self.default = None;
        }
        Some(removed)
    }

    fn position(&self, name: &str) -> Option<usize> {
        self.catalogs
            .iter()
            .position(|c| c.name == name)
            .or_else(|| {
                self.catalogs
                    .iter()
                    .position(|c| c.name.eq_ignore_ascii_case(name))
            })
    }

    /// The attached catalog called `name` (exact, else ignoring case).
    pub(crate) fn catalog_name(&self, name: &str) -> Option<&str> {
        self.position(name).map(|i| self.catalogs[i].name.as_str())
    }

    /// The attached catalog a table name explicitly starts with.
    pub(crate) fn named_catalog(&self, parts: &[String]) -> Option<&str> {
        match parts {
            [first, _, ..] => self.catalog_name(first),
            _ => None,
        }
    }

    /// The catalog table a table name refers to: a leading part naming an attached catalog
    /// selects it, else the default catalog; a name without namespace parts gets the catalog's
    /// default namespace. `None` if no catalog applies.
    pub(crate) fn resolve(&self, parts: &[String]) -> Option<CatalogTable> {
        let (catalog, rest) = self.select(parts)?;
        let identifier = match rest {
            [table] => {
                let mut identifier = catalog.default_namespace.clone()?;
                identifier.push(table.clone());
                identifier
            },
            _ => rest.to_vec(),
        };
        Some(CatalogTable {
            catalog: catalog.name.clone(),
            identifier,
        })
    }

    /// The catalog namespace a schema name refers to: a leading part naming an attached catalog
    /// selects it, else the default catalog; the other parts are the namespace (never relative
    /// to a default namespace). `None` if no catalog applies.
    pub(crate) fn resolve_namespace(&self, parts: &[String]) -> Option<CatalogNamespace> {
        let (catalog, rest) = self.select(parts)?;
        Some(CatalogNamespace {
            catalog: catalog.name.clone(),
            namespace: rest.to_vec(),
        })
    }

    /// The catalog a name selects, and the name's other parts.
    fn select<'a>(&self, parts: &'a [String]) -> Option<(&AttachedCatalog, &'a [String])> {
        let named = match parts {
            [first, rest @ ..] if !rest.is_empty() => self.position(first).map(|idx| (idx, rest)),
            _ => None,
        };
        let (idx, rest) = match named {
            Some(named) => named,
            None => (self.position(self.default.as_deref()?)?, parts),
        };
        Some((&self.catalogs[idx], rest))
    }

    /// Why a table name that nothing resolved is not a catalog table, if a catalog could have
    /// applied.
    pub(crate) fn unresolved_hint(&self, parts: &[String]) -> Option<String> {
        if self.is_empty() || self.resolve(parts).is_some() {
            return None;
        }
        let catalog = match self.named_catalog(parts) {
            Some(named) => named,
            None => match &self.default {
                Some(default) => default.as_str(),
                None => return Some("; no default catalog is set".to_string()),
            },
        };
        Some(format!("; catalog '{catalog}' has no default namespace"))
    }
}

/// The identifier parts of a table name, or `None` if a part is not an identifier.
pub(crate) fn name_parts(name: &ObjectName) -> Option<Vec<String>> {
    name.0
        .iter()
        .map(|part| part.as_ident().map(|ident| ident.value.clone()))
        .collect()
}

/// A table name as a statement wrote it, and the full name of the catalog table it refers to.
pub(crate) type RewrittenName = (Vec<String>, Vec<String>);

/// The names a statement wrote for its catalog tables, by the position of each rewritten name
/// (its first part's span start), so a table is also known by the name written there.
pub(crate) type CatalogSpellings = PlHashMap<(u64, u64), Vec<String>>;

/// The position a rewritten name is known by in `CatalogSpellings`.
pub(crate) fn name_position(name: &ObjectName) -> Option<(u64, u64)> {
    let start = name.0.first()?.as_ident()?.span.start;
    Some((start.line, start.column))
}

/// Rewrite the names of catalog tables in `stmt` to their full names, `lineitem` to
/// `onelake.tpch.lineitem`, so the rest of the translation knows them by that name. Names of
/// CTEs in scope and of local tables (`is_local`) are left alone. Returns each rewritten name
/// as written with its full name, and where each name was written.
pub(crate) fn rewrite_catalog_names(
    stmt: &mut Statement,
    resolve: impl Fn(&[String]) -> Option<Vec<String>>,
    is_local: impl Fn(&str) -> bool,
) -> PolarsResult<(Vec<RewrittenName>, CatalogSpellings)> {
    let mut registered = RegisteredAliases(Vec::new());
    let _ = dyn_visit::visit(&*stmt, &mut registered);
    let mut rewriter = Rewriter {
        resolve,
        is_local,
        registered_aliases: registered.0,
        cte_scopes: Vec::new(),
        with_stack: Vec::new(),
        rewritten: Vec::new(),
        spellings: CatalogSpellings::default(),
        desugared: 0,
    };
    if let ControlFlow::Break(err) = dyn_visit::visit_mut(stmt, &mut rewriter) {
        return Err(err);
    }
    Ok((rewriter.rewritten, rewriter.spellings))
}

/// The aliases of the statement's derived tables, table functions and `UNNEST`s, which are
/// registered as tables while the statement runs (#30070), wherever they are written.
struct RegisteredAliases(Vec<String>);

impl Visitor for RegisteredAliases {
    type Break = ();

    fn pre_visit_table_factor(&mut self, factor: &TableFactor) -> ControlFlow<()> {
        let alias = match factor {
            TableFactor::Table {
                args: Some(_),
                alias,
                ..
            }
            | TableFactor::Derived { alias, .. }
            | TableFactor::UNNEST { alias, .. } => alias.as_ref(),
            _ => None,
        };
        self.0.extend(alias.map(|alias| alias.name.value.clone()));
        ControlFlow::Continue(())
    }
}

struct Rewriter<R, L> {
    resolve: R,
    is_local: L,
    /// Aliases that tables of the same name would be looked up as (see [`RegisteredAliases`]).
    registered_aliases: Vec<String>,
    /// The CTE names visible in each enclosing query.
    cte_scopes: Vec<Vec<String>>,
    /// The `WITH` clauses of the queries being visited, taken out while their bodies are visited.
    with_stack: Vec<Option<With>>,
    rewritten: Vec<RewrittenName>,
    spellings: CatalogSpellings,
    /// The number of `TABLE x` desugared, which numbers their (made-up) name positions.
    desugared: u64,
}

impl<R, L> Rewriter<R, L>
where
    R: Fn(&[String]) -> Option<Vec<String>>,
    L: Fn(&str) -> bool,
{
    fn is_cte(&self, name: &str) -> bool {
        self.cte_scopes
            .iter()
            .flatten()
            .any(|cte| cte.eq_ignore_ascii_case(name))
    }

    /// The full name of the catalog table that `parts` refers to: not a CTE or a local table.
    fn resolve_name(&self, parts: &[String]) -> Option<Vec<String>> {
        let written = parts.join(".");
        if self.is_cte(&written) || (self.is_local)(&written) {
            return None;
        }
        (self.resolve)(parts)
    }

    /// Record `parts` as referring to the catalog table `full`. Tables are looked up by their
    /// dotted name, so it must be the catalog table's alone.
    fn record(&mut self, parts: &[String], full: &[String]) -> PolarsResult<()> {
        let key = full.join(".");
        let is_alias = self
            .registered_aliases
            .iter()
            .any(|alias| alias.eq_ignore_ascii_case(&key));
        if is_alias || self.is_cte(&key) || (self.is_local)(&key) {
            polars_bail!(
                SQLInterface: "table '{}' refers to catalog table '{}', which is also the name of a registered table, CTE or alias; rename one of them",
                display_name(parts), display_name(full)
            );
        }
        if let Some((_, other)) = self
            .rewritten
            .iter()
            .find(|(_, other)| other.join(".") == key && other != full)
        {
            polars_bail!(
                SQLInterface: "catalog tables {} and {} have the same dotted name, which SQL cannot tell apart",
                display_name(other), display_name(full)
            );
        }
        if !self.rewritten.iter().any(|(written, _)| written == parts) {
            self.rewritten.push((parts.to_vec(), full.to_vec()));
        }
        Ok(())
    }

    /// Replace `TABLE x` of a catalog table by `SELECT * FROM x`, whose table name the visit
    /// then rewrites (sqlparser's `TABLE` holds at most two name parts).
    fn desugar_table(&mut self, body: &mut SetExpr) {
        match body {
            SetExpr::SetOperation { left, right, .. } => {
                self.desugar_table(left);
                self.desugar_table(right);
            },
            SetExpr::Table(table) => {
                let parts: Vec<String> = table
                    .schema_name
                    .iter()
                    .chain(&table.table_name)
                    .cloned()
                    .collect();
                if !parts.is_empty() && self.resolve_name(&parts).is_some() {
                    // Line 0 is never a position in the query text.
                    self.desugared += 1;
                    let at = Location::new(0, self.desugared);
                    *body = select_star_from(&parts, Span::new(at, at));
                }
            },
            _ => {},
        }
    }
}

impl<R, L> VisitorMut for Rewriter<R, L>
where
    R: Fn(&[String]) -> Option<Vec<String>>,
    L: Fn(&str) -> bool,
{
    type Break = PolarsError;

    fn pre_visit_query(&mut self, query: &mut Query) -> ControlFlow<PolarsError> {
        // A non-recursive CTE is visible after its definition, as `register_ctes` registers them.
        let mut with = query.with.take();
        let mut names = Vec::new();
        if let Some(with) = with.as_mut() {
            let all: Vec<String> = with
                .cte_tables
                .iter()
                .map(|cte| cte.alias.name.value.clone())
                .collect();
            let recursive = with.recursive;
            for cte in &mut with.cte_tables {
                self.cte_scopes.push(if recursive {
                    all.clone()
                } else {
                    names.clone()
                });
                let flow = dyn_visit::visit_mut(&mut *cte.query, self);
                self.cte_scopes.pop();
                flow?;
                names.push(cte.alias.name.value.clone());
            }
        }
        self.cte_scopes.push(names);
        self.with_stack.push(with);
        self.desugar_table(&mut query.body);
        ControlFlow::Continue(())
    }

    fn post_visit_query(&mut self, query: &mut Query) -> ControlFlow<PolarsError> {
        self.cte_scopes.pop();
        query.with = self.with_stack.pop().flatten();
        ControlFlow::Continue(())
    }

    fn pre_visit_table_factor(&mut self, factor: &mut TableFactor) -> ControlFlow<PolarsError> {
        // A name with arguments is a table function.
        let TableFactor::Table {
            name,
            args: None,
            version,
            ..
        } = factor
        else {
            return ControlFlow::Continue(());
        };
        let Some(parts) = name_parts(name) else {
            return ControlFlow::Continue(());
        };
        if parts.iter().any(|part| is_version_part(part)) {
            return ControlFlow::Break(polars_err!(
                SQLInterface: "invalid table name {}", name
            ));
        }
        let written = parts.join(".");
        if self.is_cte(&written) || (self.is_local)(&written) {
            // a time travel clause is refused where the table is read
            return ControlFlow::Continue(());
        }
        let spec = match version
            .as_ref()
            .map(VersionSpec::from_table_version)
            .transpose()
        {
            Ok(spec) => spec,
            Err(err) => return ControlFlow::Break(err),
        };
        // A versioned read is its own relation, known by the name and the version.
        let key = with_version(parts.clone(), spec.as_ref());
        if let Some(full) = (self.resolve)(&key) {
            if let Err(err) = self.record(&key, &full) {
                return ControlFlow::Break(err);
            }
            let span = name.0[0]
                .as_ident()
                .map_or_else(Span::empty, |ident| ident.span);
            if let Some(position) = name_position(name) {
                self.spellings.insert(position, parts);
            }
            *name = object_name(&full, span);
            *version = None;
        }
        ControlFlow::Continue(())
    }
}

/// A table name whose parts all have `span`, so the name keeps the position it was written at.
fn object_name(parts: &[String], span: Span) -> ObjectName {
    let ident = |part: &String| Ident {
        value: part.clone(),
        quote_style: None,
        span,
    };
    ObjectName(
        parts
            .iter()
            .map(|part| ObjectNamePart::Identifier(ident(part)))
            .collect(),
    )
}

/// `SELECT * FROM <parts>`, with the name at `span`.
fn select_star_from(parts: &[String], span: Span) -> SetExpr {
    let mut stmts = Parser::parse_sql(&GenericDialect, "SELECT * FROM t").unwrap();
    let Statement::Query(query) = stmts.remove(0) else {
        unreachable!()
    };
    let mut body = *query.body;
    if let SetExpr::Select(select) = &mut body
        && let TableFactor::Table { name, .. } = &mut select.from[0].relation
    {
        *name = object_name(parts, span);
    }
    body
}

/// The schema of a catalog table from its column definitions, and its `NOT NULL` columns.
pub(crate) fn create_table_columns(columns: &[ColumnDef]) -> PolarsResult<(Schema, Vec<String>)> {
    let mut schema = Schema::with_capacity(columns.len());
    let mut required = Vec::new();
    for col in columns {
        let name = col.name.value.as_str();
        let mut not_null = None;
        for option in &col.options {
            let is_not_null = match &option.option {
                ColumnOption::NotNull => true,
                ColumnOption::Null => false,
                // the default of a nullable column anyway
                ColumnOption::Default(expr) if is_null(expr) => continue,
                _ => polars_bail!(
                    SQLInterface: "only NOT NULL constraints are supported for catalog tables; found `{}` on column '{}'",
                    option, name
                ),
            };
            polars_ensure!(
                not_null.is_none_or(|n| n == is_not_null),
                SQLInterface: "conflicting NULL/NOT NULL declarations for column '{}'", name
            );
            not_null = Some(is_not_null);
        }
        let dtype = map_sql_dtype_to_polars(&col.data_type)?;
        polars_ensure!(
            schema.insert(name.into(), dtype).is_none(),
            Duplicate: "column '{}' is defined more than once", name
        );
        if not_null == Some(true) {
            required.push(name.to_string());
        }
    }
    Ok((schema, required))
}

/// The name and type of a column added by `ALTER TABLE ... ADD COLUMN`, which is nullable.
pub(crate) fn alter_table_column(col: &ColumnDef) -> PolarsResult<(String, DataType)> {
    let name = col.name.value.as_str();
    for option in &col.options {
        match &option.option {
            ColumnOption::Null => {},
            ColumnOption::Default(expr) if is_null(expr) => {},
            ColumnOption::NotNull => polars_bail!(
                SQLInterface: "`ADD COLUMN ... NOT NULL` is not supported: the table's rows have no value for column '{}'",
                name
            ),
            _ => polars_bail!(
                SQLInterface: "`ADD COLUMN` takes only NULL or DEFAULT NULL; found `{}` on column '{}'",
                option, name
            ),
        }
    }
    Ok((name.to_string(), map_sql_dtype_to_polars(&col.data_type)?))
}

/// Whether `expr` is the literal `NULL`.
fn is_null(expr: &Expr) -> bool {
    matches!(
        expr,
        Expr::Value(ValueWithSpan {
            value: Value::Null,
            ..
        })
    )
}

/// The properties of an `ALTER TABLE ... SET TBLPROPERTIES (...)`.
pub(crate) fn alter_table_properties(options: &[SqlOption]) -> PolarsResult<Vec<(String, String)>> {
    let mut properties: Vec<(String, String)> = Vec::new();
    for option in options {
        let (key, value) = option_key_value(option, "table", &["location", "format-version"])?;
        polars_ensure!(
            !RESERVED_TABLE_PROPERTIES.contains(&key) && key != "format-version",
            SQLInterface: "table property '{}' is reserved by Iceberg", key
        );
        polars_ensure!(
            key != "location",
            SQLInterface: "a table's location is not a property and cannot be set"
        );
        polars_ensure!(
            properties.iter().all(|(k, _)| k != key),
            SQLInterface: "duplicate table property '{}'", key
        );
        properties.push((key.to_string(), value));
    }
    Ok(properties)
}

/// Iceberg's reserved table properties, other than `format-version`, which a create may set.
const RESERVED_TABLE_PROPERTIES: [&str; 8] = [
    "uuid",
    "snapshot-count",
    "current-snapshot-id",
    "current-snapshot-summary",
    "current-snapshot-timestamp-ms",
    "current-schema",
    "default-partition-spec",
    "default-sort-order",
];

/// Table properties, in the order written.
type TableProperties = Vec<(String, String)>;

/// The location and properties of a catalog table, from `LOCATION` and `WITH (...)`.
pub(crate) fn create_table_options(
    options: &CreateTableOptions,
    location_clause: Option<&str>,
) -> PolarsResult<(Option<String>, TableProperties)> {
    let options = match options {
        CreateTableOptions::None => &[][..],
        CreateTableOptions::With(options) => options.as_slice(),
        CreateTableOptions::TableProperties(_) => {
            polars_bail!(SQLInterface: "`TBLPROPERTIES` is not supported; use `WITH (...)` instead")
        },
        _ => {
            polars_bail!(SQLInterface: "table `OPTIONS` clauses are not supported; use `WITH (...)` instead")
        },
    };
    let mut location = location_clause.map(str::to_string);
    let mut properties = TableProperties::new();
    for option in options {
        let (key, value) = option_key_value(option, "table", &["location", "format-version"])?;
        polars_ensure!(
            !RESERVED_TABLE_PROPERTIES.contains(&key),
            SQLInterface: "table option '{}' is reserved by Iceberg", key
        );
        if key == "location" {
            polars_ensure!(
                location_clause.is_none(),
                SQLInterface: "`LOCATION` and table option 'location' cannot be combined"
            );
            polars_ensure!(location.is_none(), SQLInterface: "duplicate table option 'location'");
            location = Some(value);
        } else {
            polars_ensure!(
                properties.iter().all(|(k, _)| k != key),
                SQLInterface: "duplicate table option '{}'", key
            );
            properties.push((key.to_string(), value));
        }
    }
    if let Some(location) = &location {
        polars_ensure!(
            is_absolute_location(location),
            SQLInterface: "location must be an absolute path or a URI; found '{}'", location
        );
    }
    Ok((location, properties))
}

/// The partition fields of a catalog table, from `PARTITION BY (...)` or `PARTITIONED BY (...)`.
pub(crate) fn create_table_partitioning(
    partition_by: Option<&Expr>,
    hive_distribution: &HiveDistributionStyle,
) -> PolarsResult<Vec<CatalogPartitionField>> {
    // Each field with its SQL, for messages.
    let fields: Vec<(CatalogPartitionField, String)> = match (partition_by, hive_distribution) {
        (Some(_), HiveDistributionStyle::PARTITIONED { .. }) => {
            polars_bail!(SQLInterface: "`PARTITION BY` and `PARTITIONED BY` cannot be combined")
        },
        (Some(expr), _) => {
            let exprs = match unparenthesized(expr) {
                Expr::Tuple(exprs) => exprs.as_slice(),
                expr => std::slice::from_ref(expr),
            };
            partition_fields(exprs)?
        },
        (None, HiveDistributionStyle::PARTITIONED { columns }) => columns
            .iter()
            .map(|col| {
                // Hive's `PARTITIONED BY (p INT)` adds a column.
                polars_ensure!(
                    col.data_type == SQLDataType::Unspecified && col.options.is_empty(),
                    SQLInterface: "`PARTITIONED BY` takes column names only; found `{}`; types and constraints belong in the column list",
                    col
                );
                let field = CatalogPartitionField {
                    path: vec![col.name.value.clone()],
                    transform: "identity".to_string(),
                };
                Ok((field, col.name.value.clone()))
            })
            .collect::<PolarsResult<_>>()?,
        (None, _) => Vec::new(),
    };
    ensure_unique_fields(&fields)?;
    // A new spec takes one time transform per column, as Iceberg's spec builders do.
    let is_time = |f: &CatalogPartitionField| {
        matches!(f.transform.as_str(), "year" | "month" | "day" | "hour")
    };
    for (i, (field, sql)) in fields.iter().enumerate() {
        for (earlier, earlier_sql) in &fields[..i] {
            polars_ensure!(
                earlier.path != field.path || !is_time(earlier) || !is_time(field),
                SQLInterface: "partition fields `{}` and `{}` both partition column '{}' by time; use one of them",
                earlier_sql, sql, field.path.join(".")
            );
        }
    }
    Ok(fields.into_iter().map(|(field, _)| field).collect())
}

/// The partition fields of `SET PARTITIONED BY (...)`. Which time transforms may share a column
/// depends on the table's current spec, so that is checked with the table.
pub(crate) fn set_partitioning(exprs: &[Expr]) -> PolarsResult<Vec<CatalogPartitionField>> {
    let fields = partition_fields(exprs)?;
    ensure_unique_fields(&fields)?;
    Ok(fields.into_iter().map(|(field, _)| field).collect())
}

/// The partition fields of `exprs`, each with its SQL, for messages.
fn partition_fields(exprs: &[Expr]) -> PolarsResult<Vec<(CatalogPartitionField, String)>> {
    exprs
        .iter()
        .map(|expr| Ok((partition_field(expr)?, expr.to_string())))
        .collect()
}

fn ensure_unique_fields(fields: &[(CatalogPartitionField, String)]) -> PolarsResult<()> {
    for (i, (field, sql)) in fields.iter().enumerate() {
        polars_ensure!(
            fields[..i].iter().all(|(earlier, _)| earlier != field),
            SQLInterface: "duplicate partition field `{}`", sql
        );
    }
    Ok(())
}

/// A partition field: a source, `year|month|day|hour(source)` (with Iceberg's aliases), or
/// `bucket|truncate(N, source)`. A source is a column or a struct field (`s.f`).
fn partition_field(expr: &Expr) -> PolarsResult<CatalogPartitionField> {
    let unsupported = || {
        polars_err!(
            SQLInterface: "unsupported partition field `{}`; expected a column or struct field (`s.f`), or year, month, day, hour, bucket(N, ...) or truncate(W, ...) of one",
            expr
        )
    };
    // Checked against the schema later: a struct field, never a table-qualified column.
    let source_of = |expr: &Expr| match unparenthesized(expr) {
        Expr::Identifier(ident) => Some(vec![ident.value.clone()]),
        Expr::CompoundIdentifier(idents) => Some(idents.iter().map(|i| i.value.clone()).collect()),
        _ => None,
    };
    if let Some(path) = source_of(expr) {
        return Ok(CatalogPartitionField {
            path,
            transform: "identity".to_string(),
        });
    }
    let Expr::Function(Function {
        name,
        uses_odbc_syntax: false,
        parameters: FunctionArguments::None,
        args: FunctionArguments::List(list),
        filter: None,
        null_treatment: None,
        over: None,
        within_group,
    }) = unparenthesized(expr)
    else {
        return Err(unsupported());
    };
    let fname = match name_parts(name).as_deref() {
        Some([fname])
            if within_group.is_empty()
                && list.duplicate_treatment.is_none()
                && list.clauses.is_empty() =>
        {
            fname.to_ascii_lowercase()
        },
        _ => return Err(unsupported()),
    };
    let args = list
        .args
        .iter()
        .map(|arg| match arg {
            FunctionArg::Unnamed(FunctionArgExpr::Expr(expr)) => Some(expr),
            _ => None,
        })
        .collect::<Option<Vec<_>>>()
        .ok_or_else(unsupported)?;
    let positive_int = |expr: &Expr| match unparenthesized(expr) {
        Expr::Value(ValueWithSpan {
            value: Value::Number(n, false),
            ..
        }) => n.parse::<i32>().ok().filter(|n| *n > 0),
        _ => None,
    };
    let (transform, source) = match (fname.as_str(), args.as_slice()) {
        ("year" | "years", [source]) => ("year".to_string(), source),
        ("month" | "months", [source]) => ("month".to_string(), source),
        ("day" | "days", [source]) => ("day".to_string(), source),
        ("hour" | "hours", [source]) => ("hour".to_string(), source),
        ("bucket" | "truncate", [n, source]) => {
            let Some(n) = positive_int(n) else {
                if source_of(n).is_some() && positive_int(source).is_some() {
                    polars_bail!(SQLInterface: "`{}` takes the number first, as in `{}({}, {})`", expr, fname, source, n)
                }
                polars_bail!(SQLInterface: "the first argument of `{}` must be an integer from 1 to {}", expr, i32::MAX)
            };
            (format!("{fname}[{n}]"), source)
        },
        _ => return Err(unsupported()),
    };
    let path = source_of(source).ok_or_else(unsupported)?;
    Ok(CatalogPartitionField { path, transform })
}

/// `expr` without enclosing parentheses.
fn unparenthesized(mut expr: &Expr) -> &Expr {
    while let Expr::Nested(inner) = expr {
        expr = inner;
    }
    expr
}

/// The properties of a catalog namespace, from `WITH (...)`.
pub(crate) fn create_schema_options(options: &[SqlOption]) -> PolarsResult<Vec<(String, String)>> {
    let mut properties: Vec<(String, String)> = Vec::new();
    for option in options {
        let (key, value) = option_key_value(option, "schema", &["location"])?;
        polars_ensure!(
            key != "location" || is_absolute_location(&value),
            SQLInterface: "location must be an absolute path or a URI; found '{}'", value
        );
        polars_ensure!(
            properties.iter().all(|(k, _)| k != key),
            SQLInterface: "duplicate schema option '{}'", key
        );
        properties.push((key.to_string(), value));
    }
    Ok(properties)
}

/// The key and value of a `'key' = value` option of a `kind` ("table", "schema"), with keys that
/// differ from a `known` one only in case rejected.
fn option_key_value<'a>(
    option: &'a SqlOption,
    kind: &str,
    known: &[&str],
) -> PolarsResult<(&'a str, String)> {
    let SqlOption::KeyValue { key, value } = option else {
        polars_bail!(SQLInterface: "{} options must be `'key' = value` pairs; found `{}`", kind, option)
    };
    let key = key.value.as_str();
    polars_ensure!(!key.is_empty(), SQLInterface: "{} option names must not be empty", kind);
    for known in known {
        polars_ensure!(
            key == *known || !key.eq_ignore_ascii_case(known),
            SQLInterface: "'{}' is not a {} option; did you mean '{}'?", key, kind, known
        );
    }
    let Some(value) = option_value(value) else {
        polars_bail!(
            SQLInterface: "{} option values must be string, number or boolean literals; found `{}` for '{}'",
            kind, value, key
        )
    };
    Ok((key, value))
}

/// A table option value as a property string, if it is a string, number or boolean literal.
fn option_value(value: &Expr) -> Option<String> {
    match value {
        Expr::Value(v) => match &v.value {
            Value::SingleQuotedString(s) => Some(s.clone()),
            Value::Number(n, _) => Some(n.clone()),
            Value::Boolean(b) => Some(b.to_string()),
            _ => None,
        },
        Expr::UnaryOp {
            op: UnaryOperator::Minus,
            expr,
        } => match expr.as_ref() {
            Expr::Value(ValueWithSpan {
                value: Value::Number(n, _),
                ..
            }) => Some(format!("-{n}")),
            _ => None,
        },
        _ => None,
    }
}

/// Whether a location is an absolute path (POSIX or Windows) or has a URI scheme.
fn is_absolute_location(location: &str) -> bool {
    let bytes = location.as_bytes();
    let windows_path = bytes.len() > 2
        && bytes[0].is_ascii_alphabetic()
        && bytes[1] == b':'
        && matches!(bytes[2], b'/' | b'\\');
    let has_scheme = location.split_once("://").is_some_and(|(scheme, _)| {
        scheme.starts_with(|c: char| c.is_ascii_alphabetic())
            && scheme
                .chars()
                .all(|c| c.is_ascii_alphanumeric() || matches!(c, '+' | '-' | '.'))
    });
    location.starts_with('/') || windows_path || has_scheme
}

/// A table name for messages, with parts containing dots quoted.
pub(crate) fn display_name(parts: &[String]) -> String {
    let (parts, version) = split_version(parts);
    let parts: Vec<String> = parts
        .iter()
        .map(|part| match part.contains('.') {
            true => format!("\"{part}\""),
            false => part.clone(),
        })
        .collect();
    match version {
        Some(version) => format!("{} {}", parts.join("."), version),
        None => parts.join("."),
    }
}

/// The first part of a `DELETE` condition that a catalog filter can't express, described for a
/// message. Only a hint: converting the planned predicate decides.
pub(crate) fn hint_unsupported_part(expr: &Expr) -> Option<String> {
    use BinaryOperator as Op;
    if !expr_references_any_column(expr) {
        return None;
    }
    match expr {
        Expr::Nested(e)
        | Expr::UnaryOp {
            op: UnaryOperator::Not,
            expr: e,
        } => hint_unsupported_part(e),
        Expr::BinaryOp {
            left,
            op: Op::And | Op::Or,
            right,
        } => hint_unsupported_part(left).or_else(|| hint_unsupported_part(right)),
        Expr::BinaryOp {
            left,
            op: Op::Eq | Op::NotEq | Op::Lt | Op::LtEq | Op::Gt | Op::GtEq,
            right,
        } => match expr_references_any_column(left) && expr_references_any_column(right) {
            true => Some(format!("comparison of two columns `{expr}`")),
            false => unsupported_operand(left)
                .or_else(|| unsupported_operand(right))
                .or_else(|| {
                    unsupported_string(left, false).or_else(|| unsupported_string(right, false))
                }),
        },
        Expr::IsDistinctFrom(left, right) | Expr::IsNotDistinctFrom(left, right) => {
            match expr_references_any_column(left) && expr_references_any_column(right) {
                true => Some(format!("comparison of two columns `{expr}`")),
                false => unsupported_operand(left)
                    .or_else(|| unsupported_operand(right))
                    .or_else(|| {
                        unsupported_string(left, false).or_else(|| unsupported_string(right, false))
                    }),
            }
        },
        Expr::IsNull(e)
        | Expr::IsNotNull(e)
        | Expr::IsTrue(e)
        | Expr::IsNotTrue(e)
        | Expr::IsFalse(e)
        | Expr::IsNotFalse(e) => unsupported_operand(e),
        Expr::InList { expr: e, list, .. } => unsupported_operand(e).or_else(|| {
            match list.iter().any(|item| {
                matches!(
                    item,
                    Expr::Value(ValueWithSpan {
                        value: Value::Null,
                        ..
                    })
                )
            }) {
                true => Some("NULL in the IN list".to_string()),
                false => list
                    .iter()
                    .find_map(|item| column_in_bound(item, expr))
                    .or_else(|| list.iter().find_map(|item| unsupported_string(item, false))),
            }
        }),
        Expr::Between {
            expr: e, low, high, ..
        } => unsupported_operand(e)
            .or_else(|| column_in_bound(low, expr))
            .or_else(|| column_in_bound(high, expr))
            .or_else(|| unsupported_string(low, false))
            .or_else(|| unsupported_string(high, false)),
        Expr::Like {
            any,
            expr: e,
            pattern,
            escape_char,
            ..
        } => unsupported_operand(e).or_else(|| {
            let prefix = !any
                && escape_char.is_none()
                && matches!(pattern.as_ref(), Expr::Value(ValueWithSpan {
                    value: Value::SingleQuotedString(p), ..
                }) if is_prefix_pattern(p));
            match prefix {
                true => unsupported_string(pattern, true),
                false => Some(format!("LIKE pattern {pattern} (only 'prefix%')")),
            }
        }),
        Expr::ILike { .. } => Some("ILIKE".to_string()),
        Expr::Identifier(_) | Expr::CompoundIdentifier(_) => None,
        _ => Some(describe_operand(expr)),
    }
}

fn unsupported_operand(expr: &Expr) -> Option<String> {
    match expr {
        Expr::Identifier(_) | Expr::CompoundIdentifier(_) => None,
        Expr::Nested(e) => unsupported_operand(e),
        _ if !expr_references_any_column(expr) => None,
        _ => Some(describe_operand(expr)),
    }
}

/// With `like`, a trailing '%' is the pattern's wildcard, not part of the constant.
fn unsupported_string(expr: &Expr, like: bool) -> Option<String> {
    let Expr::Value(ValueWithSpan {
        value: Value::SingleQuotedString(s),
        ..
    }) = expr
    else {
        return None;
    };
    let text = match like {
        true => s.strip_suffix('%').unwrap_or(s),
        false => s,
    };
    // as `sanitize` in polars-plan's pyarrow predicates
    let supported = |c: char| c.is_alphanumeric() || matches!(c, ' ' | '-' | '_');
    (!text.chars().all(supported)).then(|| {
        format!("string constant {expr} (only letters, digits, space, '-' and '_' for now)")
    })
}

fn column_in_bound(bound: &Expr, expr: &Expr) -> Option<String> {
    expr_references_any_column(bound).then(|| format!("comparison of two columns `{expr}`"))
}

fn describe_operand(expr: &Expr) -> String {
    use BinaryOperator as Op;
    match expr {
        Expr::Function(_) => format!("function `{expr}`"),
        Expr::Cast { .. } => format!("cast `{expr}`"),
        Expr::BinaryOp {
            op: Op::Plus | Op::Minus | Op::Multiply | Op::Divide | Op::Modulo,
            ..
        }
        | Expr::UnaryOp {
            op: UnaryOperator::Minus | UnaryOperator::Plus,
            ..
        } => format!("arithmetic `{expr}`"),
        _ => format!("expression `{expr}`"),
    }
}

fn is_prefix_pattern(pattern: &str) -> bool {
    match pattern.strip_suffix('%') {
        Some(prefix) => !prefix.is_empty() && !prefix.contains(['%', '_']),
        None => !pattern.contains(['%', '_']),
    }
}
