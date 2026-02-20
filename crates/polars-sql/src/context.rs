use std::ops::Deref;
use std::sync::RwLock;

use polars_core::frame::row::Row;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_ops::frame::{JoinCoalesce, MaintainOrderJoin};
use polars_plan::dsl::function_expr::StructFunction;
use polars_plan::prelude::*;
use polars_utils::aliases::{PlHashSet, PlIndexSet};
use polars_utils::format_pl_smallstr;
use sqlparser::ast::{
    BinaryOperator, CreateTable, CreateTableLikeKind, Delete, Distinct, ExcludeSelectItem,
    Expr as SQLExpr, Fetch, FromTable, FunctionArg, GroupByExpr, Ident, JoinConstraint,
    JoinOperator, LimitClause, NamedWindowDefinition, NamedWindowExpr, ObjectName, ObjectType,
    OrderBy, OrderByKind, Query, RenameSelectItem, Select, SelectFlavor, SelectItem,
    SelectItemQualifiedWildcardKind, SetExpr, SetOperator, SetQuantifier, Statement, TableAlias,
    TableFactor, TableWithJoins, Truncate, UnaryOperator, Value as SQLValue, ValueWithSpan, Values,
    Visit, WildcardAdditionalOptions, WindowSpec,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::function_registry::{DefaultFunctionRegistry, FunctionRegistry};
use crate::sql_expr::{
    parse_sql_array, parse_sql_expr, resolve_compound_identifier, to_sql_interface_err,
};
use crate::sql_visitors::{
    QualifyExpression, TableIdentifierCollector, check_for_ambiguous_column_refs,
    expr_has_window_functions, expr_refers_to_table,
};
use crate::table_functions::PolarsTableFunctions;
use crate::types::map_sql_dtype_to_polars;

#[derive(Clone)]
pub struct TableInfo {
    pub(crate) frame: LazyFrame,
    pub(crate) name: PlSmallStr,
    pub(crate) schema: Arc<Schema>,
}

struct SelectModifiers {
    exclude: PlHashSet<String>,                // SELECT * EXCLUDE
    ilike: Option<regex::Regex>,               // SELECT * ILIKE
    rename: PlHashMap<PlSmallStr, PlSmallStr>, // SELECT * RENAME
    replace: Vec<Expr>,                        // SELECT * REPLACE
}
impl SelectModifiers {
    fn matches_ilike(&self, s: &str) -> bool {
        match &self.ilike {
            Some(rx) => rx.is_match(s),
            None => true,
        }
    }
    fn renamed_cols(&self) -> Vec<Expr> {
        self.rename
            .iter()
            .map(|(before, after)| col(before.clone()).alias(after.clone()))
            .collect()
    }
}

/// For SELECT projection items; helps simplify any required disambiguation.
enum ProjectionItem {
    QualifiedExprs(PlSmallStr, Vec<Expr>),
    Exprs(Vec<Expr>),
}

/// Extract the output column name from an expression (if it has one).
fn expr_output_name(expr: &Expr) -> Option<&PlSmallStr> {
    match expr {
        Expr::Column(name) | Expr::Alias(_, name) => Some(name),
        _ => None,
    }
}

/// Disambiguate qualified wildcard columns that conflict with each other or other projections.
fn disambiguate_projection_cols(
    items: Vec<ProjectionItem>,
    schema: &Schema,
) -> PolarsResult<Vec<Expr>> {
    // Establish qualified wildcard names (with counts), and other expression names
    let mut qualified_wildcard_names: PlHashMap<PlSmallStr, usize> = PlHashMap::new();
    let mut other_names: PlHashSet<PlSmallStr> = PlHashSet::new();
    for item in &items {
        match item {
            ProjectionItem::QualifiedExprs(_, exprs) => {
                for expr in exprs {
                    if let Some(name) = expr_output_name(expr) {
                        *qualified_wildcard_names.entry(name.clone()).or_insert(0) += 1;
                    }
                }
            },
            ProjectionItem::Exprs(exprs) => {
                for expr in exprs {
                    if let Some(name) = expr_output_name(expr) {
                        other_names.insert(name.clone());
                    }
                }
            },
        }
    }

    // Names requiring disambiguation (duplicates across wildcards, eg: `tbl1.*`,`tbl2.*`)
    let needs_suffix: PlHashSet<PlSmallStr> = qualified_wildcard_names
        .into_iter()
        .filter(|(name, count)| *count > 1 || other_names.contains(name))
        .map(|(name, _)| name)
        .collect();

    // Output, applying suffixes where needed
    let mut result: Vec<Expr> = Vec::new();
    for item in items {
        match item {
            ProjectionItem::QualifiedExprs(tbl_name, exprs) if !needs_suffix.is_empty() => {
                for expr in exprs {
                    if let Some(name) = expr_output_name(&expr) {
                        if needs_suffix.contains(name) {
                            let suffixed = format_pl_smallstr!("{}:{}", name, tbl_name);
                            if schema.contains(suffixed.as_str()) {
                                result.push(col(suffixed));
                                continue;
                            }
                            if other_names.contains(name) {
                                polars_bail!(
                                    SQLInterface:
                                    "column '{}' is duplicated in the SELECT (explicitly, and via the `*` wildcard)", name
                                );
                            }
                        }
                    }
                    result.push(expr);
                }
            },
            ProjectionItem::QualifiedExprs(_, exprs) | ProjectionItem::Exprs(exprs) => {
                result.extend(exprs);
            },
        }
    }
    Ok(result)
}

/// The SQLContext is the main entry point for executing SQL queries.
#[derive(Clone)]
pub struct SQLContext {
    pub(crate) table_map: Arc<RwLock<PlHashMap<String, LazyFrame>>>,
    pub(crate) function_registry: Arc<dyn FunctionRegistry>,
    pub(crate) lp_arena: Arena<IR>,
    pub(crate) expr_arena: Arena<AExpr>,

    cte_map: PlHashMap<String, LazyFrame>,
    table_aliases: PlHashMap<String, String>,
    joined_aliases: PlHashMap<String, PlHashMap<String, String>>,
    pub(crate) named_windows: PlHashMap<String, WindowSpec>,
}

impl Default for SQLContext {
    fn default() -> Self {
        Self {
            function_registry: Arc::new(DefaultFunctionRegistry {}),
            table_map: Default::default(),
            cte_map: Default::default(),
            table_aliases: Default::default(),
            joined_aliases: Default::default(),
            named_windows: Default::default(),
            lp_arena: Default::default(),
            expr_arena: Default::default(),
        }
    }
}

impl SQLContext {
    /// Create a new SQLContext.
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # fn main() {
    /// let ctx = SQLContext::new();
    /// # }
    /// ```
    pub fn new() -> Self {
        Self::default()
    }

    /// Get the names of all registered tables, in sorted order.
    pub fn get_tables(&self) -> Vec<String> {
        let mut tables = Vec::from_iter(self.table_map.read().unwrap().keys().cloned());
        tables.sort_unstable();
        tables
    }

    /// Register a [`LazyFrame`] as a table in the SQLContext.
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// # fn main() {
    ///
    /// let mut ctx = SQLContext::new();
    /// let df = df! {
    ///    "a" =>  [1, 2, 3],
    /// }.unwrap().lazy();
    ///
    /// ctx.register("df", df);
    /// # }
    ///```
    pub fn register(&self, name: &str, lf: LazyFrame) {
        self.table_map.write().unwrap().insert(name.to_owned(), lf);
    }

    /// Unregister a [`LazyFrame`] table from the [`SQLContext`].
    pub fn unregister(&self, name: &str) {
        self.table_map.write().unwrap().remove(&name.to_owned());
    }

    /// Execute a SQL query, returning a [`LazyFrame`].
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # use polars_core::prelude::*;
    /// # use polars_lazy::prelude::*;
    /// # fn main() {
    ///
    /// let mut ctx = SQLContext::new();
    /// let df = df! {
    ///    "a" =>  [1, 2, 3],
    /// }
    /// .unwrap();
    ///
    /// ctx.register("df", df.clone().lazy());
    /// let sql_df = ctx.execute("SELECT * FROM df").unwrap().collect().unwrap();
    /// assert!(sql_df.equals(&df));
    /// # }
    ///```
    pub fn execute(&mut self, query: &str) -> PolarsResult<LazyFrame> {
        let mut parser = Parser::new(&GenericDialect);
        parser = parser.with_options(ParserOptions {
            trailing_commas: true,
            ..Default::default()
        });

        let ast = parser
            .try_with_sql(query)
            .map_err(to_sql_interface_err)?
            .parse_statements()
            .map_err(to_sql_interface_err)?;

        polars_ensure!(ast.len() == 1, SQLInterface: "one (and only one) statement can be parsed at a time");
        let res = self.execute_statement(ast.first().unwrap())?;

        // Ensure the result uses the proper arenas.
        // This will instantiate new arenas with a new version.
        let lp_arena = std::mem::take(&mut self.lp_arena);
        let expr_arena = std::mem::take(&mut self.expr_arena);
        res.set_cached_arena(lp_arena, expr_arena);

        // Every execution should clear the statement-level maps.
        self.cte_map.clear();
        self.table_aliases.clear();
        self.joined_aliases.clear();
        self.named_windows.clear();

        Ok(res)
    }

    /// Add a function registry to the SQLContext.
    /// The registry provides the ability to add custom functions to the SQLContext.
    pub fn with_function_registry(mut self, function_registry: Arc<dyn FunctionRegistry>) -> Self {
        self.function_registry = function_registry;
        self
    }

    /// Get the function registry of the SQLContext
    pub fn registry(&self) -> &Arc<dyn FunctionRegistry> {
        &self.function_registry
    }

    /// Get a mutable reference to the function registry of the SQLContext
    pub fn registry_mut(&mut self) -> &mut dyn FunctionRegistry {
        Arc::get_mut(&mut self.function_registry).unwrap()
    }
}

impl SQLContext {
    fn isolated(&self) -> Self {
        Self {
            // Deep clone to isolate
            table_map: Arc::new(RwLock::new(self.table_map.read().unwrap().clone())),
            named_windows: self.named_windows.clone(),
            cte_map: self.cte_map.clone(),

            ..Default::default()
        }
    }

    pub(crate) fn execute_statement(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        let ast = stmt;
        Ok(match ast {
            Statement::Query(query) => self.execute_query(query)?,
            stmt @ Statement::ShowTables { .. } => self.execute_show_tables(stmt)?,
            stmt @ Statement::CreateTable { .. } => self.execute_create_table(stmt)?,
            stmt @ Statement::Drop {
                object_type: ObjectType::Table,
                ..
            } => self.execute_drop_table(stmt)?,
            stmt @ Statement::Explain { .. } => self.execute_explain(stmt)?,
            stmt @ Statement::Truncate { .. } => self.execute_truncate_table(stmt)?,
            stmt @ Statement::Delete { .. } => self.execute_delete_from_table(stmt)?,
            _ => polars_bail!(
                SQLInterface: "statement type is not supported:\n{:?}", ast,
            ),
        })
    }

    pub(crate) fn execute_query(&mut self, query: &Query) -> PolarsResult<LazyFrame> {
        self.register_ctes(query)?;
        self.execute_query_no_ctes(query)
    }

    pub(crate) fn execute_query_no_ctes(&mut self, query: &Query) -> PolarsResult<LazyFrame> {
        self.validate_query(query)?;

        let lf = self.process_query(&query.body, query)?;
        self.process_limit_offset(lf, &query.limit_clause, &query.fetch)
    }

    pub(crate) fn get_frame_schema(&mut self, frame: &mut LazyFrame) -> PolarsResult<SchemaRef> {
        frame.schema_with_arenas(&mut self.lp_arena, &mut self.expr_arena)
    }

    pub(super) fn get_table_from_current_scope(&self, name: &str) -> Option<LazyFrame> {
        // Resolve the table name in the current scope; multi-stage fallback
        // * table name → cte name
        // * table alias → cte alias
        self.table_map
            .read()
            .unwrap()
            .get(name)
            .cloned()
            .or_else(|| self.cte_map.get(name).cloned())
            .or_else(|| {
                self.table_aliases.get(name).and_then(|alias| {
                    self.table_map
                        .read()
                        .unwrap()
                        .get(alias.as_str())
                        .or_else(|| self.cte_map.get(alias.as_str()))
                        .cloned()
                })
            })
    }

    /// Execute a query in an isolated context. This prevents subqueries from mutating
    /// arenas and other context state. Returns both the LazyFrame *and* its associated
    /// Schema (so that the correct arenas are used when determining schema).
    pub(crate) fn execute_isolated<F>(&mut self, query: F) -> PolarsResult<LazyFrame>
    where
        F: FnOnce(&mut Self) -> PolarsResult<LazyFrame>,
    {
        let mut ctx = self.isolated();

        // Execute query with clean state (eg: nested/subquery)
        let lf = query(&mut ctx)?;

        // Save state
        lf.set_cached_arena(ctx.lp_arena, ctx.expr_arena);

        Ok(lf)
    }

    fn expr_or_ordinal(
        &mut self,
        e: &SQLExpr,
        exprs: &[Expr],
        selected: Option<&[Expr]>,
        schema: Option<&Schema>,
        clause: &str,
    ) -> PolarsResult<Expr> {
        match e {
            SQLExpr::UnaryOp {
                op: UnaryOperator::Minus,
                expr,
            } if matches!(
                **expr,
                SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(_, _),
                    ..
                })
            ) =>
            {
                if let SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(ref idx, _),
                    ..
                }) = **expr
                {
                    Err(polars_err!(
                    SQLSyntax:
                    "negative ordinal values are invalid for {}; found -{}",
                    clause,
                    idx
                    ))
                } else {
                    unreachable!()
                }
            },
            SQLExpr::Value(ValueWithSpan {
                value: SQLValue::Number(idx, _),
                ..
            }) => {
                // note: sql queries are 1-indexed
                let idx = idx.parse::<usize>().map_err(|_| {
                    polars_err!(
                        SQLSyntax:
                        "negative ordinal values are invalid for {}; found {}",
                        clause,
                        idx
                    )
                })?;
                // note: "selected" cols represent final projection order, so we use those for
                // ordinal resolution. "exprs" may include cols that are subsequently dropped.
                let cols = if let Some(cols) = selected {
                    cols
                } else {
                    exprs
                };
                Ok(cols
                    .get(idx - 1)
                    .ok_or_else(|| {
                        polars_err!(
                            SQLInterface:
                            "{} ordinal value must refer to a valid column; found {}",
                            clause,
                            idx
                        )
                    })?
                    .clone())
            },
            SQLExpr::Value(v) => Err(polars_err!(
                SQLSyntax:
                "{} requires a valid expression or positive ordinal; found {}", clause, v,
            )),
            _ => {
                // Handle qualified cross-aliasing in ORDER BY clauses
                // (eg: `SELECT a AS b, -b AS a ... ORDER BY self.a`)
                let mut expr = parse_sql_expr(e, self, schema)?;
                if matches!(e, SQLExpr::CompoundIdentifier(_)) {
                    if let Some(schema) = schema {
                        expr = expr.map_expr(|ex| match &ex {
                            Expr::Column(name) => {
                                let prefixed = format!("__POLARS_ORIG_{}", name.as_str());
                                if schema.contains(prefixed.as_str()) {
                                    col(prefixed)
                                } else {
                                    ex
                                }
                            },
                            _ => ex,
                        });
                    }
                }
                Ok(expr)
            },
        }
    }

    pub(super) fn resolve_name(&self, tbl_name: &str, column_name: &str) -> String {
        if let Some(aliases) = self.joined_aliases.get(tbl_name) {
            if let Some(name) = aliases.get(column_name) {
                return name.to_string();
            }
        }
        column_name.to_string()
    }

    fn process_query(&mut self, expr: &SetExpr, query: &Query) -> PolarsResult<LazyFrame> {
        match expr {
            SetExpr::Select(select_stmt) => self.execute_select(select_stmt, query),
            SetExpr::Query(nested_query) => {
                let lf = self.execute_query_no_ctes(nested_query)?;
                self.process_order_by(lf, &query.order_by, None)
            },
            SetExpr::SetOperation {
                op: SetOperator::Union,
                set_quantifier,
                left,
                right,
            } => self.process_union(left, right, set_quantifier, query),

            #[cfg(feature = "semi_anti_join")]
            SetExpr::SetOperation {
                op: SetOperator::Intersect | SetOperator::Except,
                set_quantifier,
                left,
                right,
            } => self.process_except_intersect(left, right, set_quantifier, query),

            SetExpr::Values(Values {
                explicit_row: _,
                rows,
                value_keyword: _,
            }) => self.process_values(rows),

            SetExpr::Table(tbl) => {
                if let Some(table_name) = tbl.table_name.as_ref() {
                    self.get_table_from_current_scope(table_name)
                        .ok_or_else(|| {
                            polars_err!(
                                SQLInterface: "no table or alias named '{}' found",
                                tbl
                            )
                        })
                } else {
                    polars_bail!(SQLInterface: "'TABLE' requires valid table name")
                }
            },
            op => {
                polars_bail!(SQLInterface: "'{}' operation is currently unsupported", op)
            },
        }
    }

    #[cfg(feature = "semi_anti_join")]
    fn process_except_intersect(
        &mut self,
        left: &SetExpr,
        right: &SetExpr,
        quantifier: &SetQuantifier,
        query: &Query,
    ) -> PolarsResult<LazyFrame> {
        let (join_type, op_name) = match *query.body {
            SetExpr::SetOperation {
                op: SetOperator::Except,
                ..
            } => (JoinType::Anti, "EXCEPT"),
            _ => (JoinType::Semi, "INTERSECT"),
        };

        // Note: each side of the EXCEPT/INTERSECT operation should execute
        // in isolation to prevent context state leakage between them
        let mut lf = self.execute_isolated(|ctx| ctx.process_query(left, query))?;
        let mut rf = self.execute_isolated(|ctx| ctx.process_query(right, query))?;
        let lf_schema = self.get_frame_schema(&mut lf)?;

        let lf_cols: Vec<_> = lf_schema.iter_names_cloned().map(col).collect();
        let rf_cols = match quantifier {
            SetQuantifier::ByName => None,
            SetQuantifier::Distinct | SetQuantifier::None => {
                let rf_schema = self.get_frame_schema(&mut rf)?;
                let rf_cols: Vec<_> = rf_schema.iter_names_cloned().map(col).collect();
                if lf_cols.len() != rf_cols.len() {
                    polars_bail!(SQLInterface: "{} requires equal number of columns in each table (use '{} BY NAME' to combine mismatched tables)", op_name, op_name)
                }
                Some(rf_cols)
            },
            _ => {
                polars_bail!(SQLInterface: "'{} {}' is not supported", op_name, quantifier.to_string())
            },
        };
        let join = lf.join_builder().with(rf).how(join_type).join_nulls(true);
        let joined_tbl = match rf_cols {
            Some(rf_cols) => join.left_on(lf_cols).right_on(rf_cols).finish(),
            None => join.on(lf_cols).finish(),
        };
        let lf = joined_tbl.unique(None, UniqueKeepStrategy::Any);
        self.process_order_by(lf, &query.order_by, None)
    }

    fn process_union(
        &mut self,
        left: &SetExpr,
        right: &SetExpr,
        quantifier: &SetQuantifier,
        query: &Query,
    ) -> PolarsResult<LazyFrame> {
        let quantifier = *quantifier;

        // Note: each side of the UNION operation should execute
        // in isolation to prevent context state leakage between them
        let mut lf = self.execute_isolated(|ctx| ctx.process_query(left, query))?;
        let mut rf = self.execute_isolated(|ctx| ctx.process_query(right, query))?;

        let opts = UnionArgs {
            parallel: true,
            to_supertypes: true,
            maintain_order: false,
            ..Default::default()
        };
        let lf = match quantifier {
            // UNION [ALL | DISTINCT]
            SetQuantifier::All | SetQuantifier::Distinct | SetQuantifier::None => {
                let lf_schema = self.get_frame_schema(&mut lf)?;
                let rf_schema = self.get_frame_schema(&mut rf)?;
                if lf_schema.len() != rf_schema.len() {
                    polars_bail!(SQLInterface: "UNION requires equal number of columns in each table (use 'UNION BY NAME' to combine mismatched tables)")
                }
                // rename `rf` columns to match `lf` if they differ; SQL behaves
                // positionally on UNION ops (unless using the "BY NAME" qualifier)
                if lf_schema.iter_names().ne(rf_schema.iter_names()) {
                    rf = rf.rename(rf_schema.iter_names(), lf_schema.iter_names(), true);
                }
                let concatenated = concat(vec![lf, rf], opts);
                match quantifier {
                    SetQuantifier::Distinct | SetQuantifier::None => {
                        concatenated.map(|lf| lf.unique(None, UniqueKeepStrategy::Any))
                    },
                    _ => concatenated,
                }
            },
            // UNION ALL BY NAME
            #[cfg(feature = "diagonal_concat")]
            SetQuantifier::AllByName => concat_lf_diagonal(vec![lf, rf], opts),
            // UNION [DISTINCT] BY NAME
            #[cfg(feature = "diagonal_concat")]
            SetQuantifier::ByName | SetQuantifier::DistinctByName => {
                let concatenated = concat_lf_diagonal(vec![lf, rf], opts);
                concatenated.map(|lf| lf.unique(None, UniqueKeepStrategy::Any))
            },
            #[allow(unreachable_patterns)]
            _ => {
                polars_bail!(SQLInterface: "'UNION {}' is not currently supported", quantifier)
            },
        }?;

        self.process_order_by(lf, &query.order_by, None)
    }

    /// Process UNNEST as a lateral operation when it contains column references
    /// (handles `CROSS JOIN UNNEST(col) AS name` by exploding the referenced col).
    fn process_unnest_lateral(
        &self,
        lf: LazyFrame,
        alias: &Option<TableAlias>,
        array_exprs: &[SQLExpr],
        with_offset: bool,
    ) -> PolarsResult<LazyFrame> {
        let alias = alias
            .as_ref()
            .ok_or_else(|| polars_err!(SQLSyntax: "UNNEST table must have an alias"))?;
        polars_ensure!(!with_offset, SQLInterface: "UNNEST tables do not (yet) support WITH ORDINALITY|OFFSET");

        let (mut explode_cols, mut rename_from, mut rename_to) = (
            Vec::with_capacity(array_exprs.len()),
            Vec::with_capacity(array_exprs.len()),
            Vec::with_capacity(array_exprs.len()),
        );
        let is_single_col = array_exprs.len() == 1;

        for (i, arr_expr) in array_exprs.iter().enumerate() {
            let col_name = match arr_expr {
                SQLExpr::Identifier(ident) => PlSmallStr::from_str(&ident.value),
                SQLExpr::CompoundIdentifier(parts) => {
                    PlSmallStr::from_str(&parts.last().unwrap().value)
                },
                SQLExpr::Array(_) => polars_bail!(
                    SQLInterface: "CROSS JOIN UNNEST with both literal arrays and column references is not supported"
                ),
                other => polars_bail!(
                    SQLSyntax: "UNNEST expects column references or array literals, found {:?}", other
                ),
            };
            // alias: column name from "AS t(col)", or table alias
            if let Some(name) = alias
                .columns
                .get(i)
                .map(|c| c.name.value.as_str())
                .or_else(|| is_single_col.then_some(alias.name.value.as_str()))
                .filter(|name| !name.is_empty() && *name != col_name.as_str())
            {
                rename_from.push(col_name.clone());
                rename_to.push(PlSmallStr::from_str(name));
            }
            explode_cols.push(col_name);
        }

        let mut lf = lf.explode(
            Selector::ByName {
                names: Arc::from(explode_cols),
                strict: true,
            },
            ExplodeOptions {
                empty_as_null: true,
                keep_nulls: true,
            },
        );
        if !rename_from.is_empty() {
            lf = lf.rename(rename_from, rename_to, true);
        }
        Ok(lf)
    }

    fn process_values(&mut self, values: &[Vec<SQLExpr>]) -> PolarsResult<LazyFrame> {
        let frame_rows: Vec<Row> = values.iter().map(|row| {
            let row_data: Result<Vec<_>, _> = row.iter().map(|expr| {
                let expr = parse_sql_expr(expr, self, None)?;
                match expr {
                    Expr::Literal(value) => {
                        value.to_any_value()
                            .ok_or_else(|| polars_err!(SQLInterface: "invalid literal value: {:?}", value))
                            .map(|av| av.into_static())
                    },
                    _ => polars_bail!(SQLInterface: "VALUES clause expects literals; found {}", expr),
                }
            }).collect();
            row_data.map(Row::new)
        }).collect::<Result<_, _>>()?;

        Ok(DataFrame::from_rows(frame_rows.as_ref())?.lazy())
    }

    // EXPLAIN SELECT * FROM DF
    fn execute_explain(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        match stmt {
            Statement::Explain { statement, .. } => {
                let lf = self.execute_statement(statement)?;
                let plan = lf.describe_optimized_plan()?;
                let plan = plan
                    .split('\n')
                    .collect::<Series>()
                    .with_name(PlSmallStr::from_static("Logical Plan"))
                    .into_column();
                let df = DataFrame::new_infer_height(vec![plan])?;
                Ok(df.lazy())
            },
            _ => polars_bail!(SQLInterface: "unexpected statement type; expected EXPLAIN"),
        }
    }

    // SHOW TABLES
    fn execute_show_tables(&mut self, _: &Statement) -> PolarsResult<LazyFrame> {
        let tables = Column::new("name".into(), self.get_tables());
        let df = DataFrame::new_infer_height(vec![tables])?;
        Ok(df.lazy())
    }

    // DROP TABLE <tbl>
    fn execute_drop_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        match stmt {
            Statement::Drop { names, .. } => {
                names.iter().for_each(|name| {
                    self.table_map.write().unwrap().remove(&name.to_string());
                });
                Ok(DataFrame::empty().lazy())
            },
            _ => polars_bail!(SQLInterface: "unexpected statement type; expected DROP"),
        }
    }

    // DELETE FROM <tbl> [WHERE ...]
    fn execute_delete_from_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        if let Statement::Delete(Delete {
            tables,
            from,
            using,
            selection,
            returning,
            order_by,
            limit,
            delete_token: _,
        }) = stmt
        {
            let error_message: Option<&'static str> = if !tables.is_empty() {
                Some("DELETE expects exactly one table name")
            } else if using.is_some() {
                Some("DELETE does not support the USING clause")
            } else if returning.is_some() {
                Some("DELETE does not support the RETURNING clause")
            } else if limit.is_some() {
                Some("DELETE does not support the LIMIT clause")
            } else if !order_by.is_empty() {
                Some("DELETE does not support the ORDER BY clause")
            } else {
                None
            };

            if let Some(msg) = error_message {
                polars_bail!(SQLInterface: msg);
            }

            let from_tables = match &from {
                FromTable::WithFromKeyword(from) => from,
                FromTable::WithoutKeyword(from) => from,
            };
            if from_tables.len() > 1 {
                polars_bail!(SQLInterface: "cannot have multiple tables in DELETE FROM (found {})", from_tables.len())
            }
            let tbl_expr = from_tables.first().unwrap();
            if !tbl_expr.joins.is_empty() {
                polars_bail!(SQLInterface: "DELETE does not support table JOINs")
            }
            let (_, lf) = self.get_table(&tbl_expr.relation)?;
            if selection.is_none() {
                // no WHERE clause; equivalent to TRUNCATE (drop all rows)
                Ok(lf.clear())
            } else {
                // apply constraint as inverted filter (drops rows matching the selection)
                Ok(self.process_where(lf.clone(), selection, true, None)?)
            }
        } else {
            polars_bail!(SQLInterface: "unexpected statement type; expected DELETE")
        }
    }

    // TRUNCATE <tbl>
    fn execute_truncate_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        if let Statement::Truncate(Truncate {
            table_names,
            partitions,
            ..
        }) = stmt
        {
            match partitions {
                None => {
                    if table_names.len() != 1 {
                        polars_bail!(SQLInterface: "TRUNCATE expects exactly one table name; found {}", table_names.len())
                    }
                    let tbl = table_names[0].name.to_string();
                    if let Some(lf) = self.table_map.write().unwrap().get_mut(&tbl) {
                        *lf = lf.clone().clear();
                        Ok(lf.clone())
                    } else {
                        polars_bail!(SQLInterface: "table '{}' does not exist", tbl);
                    }
                },
                _ => {
                    polars_bail!(SQLInterface: "TRUNCATE does not support use of 'partitions'")
                },
            }
        } else {
            polars_bail!(SQLInterface: "unexpected statement type; expected TRUNCATE")
        }
    }

    fn register_cte(&mut self, name: &str, lf: LazyFrame) {
        self.cte_map.insert(name.to_owned(), lf);
    }

    fn register_ctes(&mut self, query: &Query) -> PolarsResult<()> {
        if let Some(with) = &query.with {
            if with.recursive {
                polars_bail!(SQLInterface: "recursive CTEs are not supported")
            }
            for cte in &with.cte_tables {
                let cte_name = cte.alias.name.value.clone();
                let mut lf = self.execute_query(&cte.query)?;
                lf = self.rename_columns_from_table_alias(lf, &cte.alias)?;
                self.register_cte(&cte_name, lf);
            }
        }
        Ok(())
    }

    fn register_named_windows(
        &mut self,
        named_windows: &[NamedWindowDefinition],
    ) -> PolarsResult<()> {
        for NamedWindowDefinition(name, expr) in named_windows {
            let spec = match expr {
                NamedWindowExpr::NamedWindow(ref_name) => self
                    .named_windows
                    .get(&ref_name.value)
                    .ok_or_else(|| {
                        polars_err!(
                            SQLInterface:
                            "named window '{}' references undefined window '{}'",
                            name.value, ref_name.value
                        )
                    })?
                    .clone(),
                NamedWindowExpr::WindowSpec(spec) => spec.clone(),
            };
            self.named_windows.insert(name.value.clone(), spec);
        }
        Ok(())
    }

    /// execute the 'FROM' part of the query
    fn execute_from_statement(&mut self, tbl_expr: &TableWithJoins) -> PolarsResult<LazyFrame> {
        let (l_name, mut lf) = self.get_table(&tbl_expr.relation)?;
        if !tbl_expr.joins.is_empty() {
            for join in &tbl_expr.joins {
                // Handle "CROSS JOIN UNNEST(col)" as a lateral join op
                if let (
                    JoinOperator::CrossJoin(JoinConstraint::None),
                    TableFactor::UNNEST {
                        alias,
                        array_exprs,
                        with_offset,
                        ..
                    },
                ) = (&join.join_operator, &join.relation)
                {
                    if array_exprs.iter().any(|e| !matches!(e, SQLExpr::Array(_))) {
                        lf = self.process_unnest_lateral(lf, alias, array_exprs, *with_offset)?;
                        continue;
                    }
                }

                let (r_name, mut rf) = self.get_table(&join.relation)?;
                if r_name.is_empty() {
                    // Require non-empty to avoid duplicate column errors from nested self-joins.
                    polars_bail!(
                        SQLInterface:
                        "cannot JOIN on unnamed relation; please provide an alias"
                    )
                }
                let left_schema = self.get_frame_schema(&mut lf)?;
                let right_schema = self.get_frame_schema(&mut rf)?;

                lf = match &join.join_operator {
                    op @ (JoinOperator::Join(constraint)  // note: bare "join" is inner
                    | JoinOperator::FullOuter(constraint)
                    | JoinOperator::Left(constraint)
                    | JoinOperator::LeftOuter(constraint)
                    | JoinOperator::Right(constraint)
                    | JoinOperator::RightOuter(constraint)
                    | JoinOperator::Inner(constraint)
                    | JoinOperator::Anti(constraint)
                    | JoinOperator::Semi(constraint)
                    | JoinOperator::LeftAnti(constraint)
                    | JoinOperator::LeftSemi(constraint)
                    | JoinOperator::RightAnti(constraint)
                    | JoinOperator::RightSemi(constraint)) => {
                        let (lf, rf) = match op {
                            JoinOperator::RightAnti(_) | JoinOperator::RightSemi(_) => (rf, lf),
                            _ => (lf, rf),
                        };
                        self.process_join(
                            &TableInfo {
                                frame: lf,
                                name: (&l_name).into(),
                                schema: left_schema.clone(),
                            },
                            &TableInfo {
                                frame: rf,
                                name: (&r_name).into(),
                                schema: right_schema.clone(),
                            },
                            constraint,
                            match op {
                                JoinOperator::Join(_) | JoinOperator::Inner(_) => JoinType::Inner,
                                JoinOperator::Left(_) | JoinOperator::LeftOuter(_) => {
                                    JoinType::Left
                                },
                                JoinOperator::Right(_) | JoinOperator::RightOuter(_) => {
                                    JoinType::Right
                                },
                                JoinOperator::FullOuter(_) => JoinType::Full,
                                #[cfg(feature = "semi_anti_join")]
                                JoinOperator::Anti(_)
                                | JoinOperator::LeftAnti(_)
                                | JoinOperator::RightAnti(_) => JoinType::Anti,
                                #[cfg(feature = "semi_anti_join")]
                                JoinOperator::Semi(_)
                                | JoinOperator::LeftSemi(_)
                                | JoinOperator::RightSemi(_) => JoinType::Semi,
                                join_type => polars_bail!(
                                    SQLInterface:
                                    "join type '{:?}' not currently supported",
                                    join_type
                                ),
                            },
                        )?
                    },
                    JoinOperator::CrossJoin(JoinConstraint::None) => {
                        lf.cross_join(rf, Some(format_pl_smallstr!(":{}", r_name)))
                    },
                    JoinOperator::CrossJoin(constraint) => {
                        polars_bail!(
                            SQLInterface:
                            "CROSS JOIN does not support {:?} constraint; consider INNER JOIN instead",
                            constraint
                        )
                    },
                    join_type => {
                        polars_bail!(SQLInterface: "join type '{:?}' not currently supported", join_type)
                    },
                };

                // track join-aliased columns so we can resolve/check them later
                let joined_schema = self.get_frame_schema(&mut lf)?;

                self.joined_aliases.insert(
                    r_name.clone(),
                    right_schema
                        .iter_names()
                        .filter_map(|name| {
                            // col exists in both tables and is aliased in the joined result
                            let aliased_name = format!("{name}:{r_name}");
                            if left_schema.contains(name)
                                && joined_schema.contains(aliased_name.as_str())
                            {
                                Some((name.to_string(), aliased_name))
                            } else {
                                None
                            }
                        })
                        .collect::<PlHashMap<String, String>>(),
                );
            }
        };
        Ok(lf)
    }

    /// Check that the SELECT statement only contains supported clauses.
    fn validate_select(&self, select_stmt: &Select) -> PolarsResult<()> {
        // Destructure "Select" exhaustively; that way if/when new fields are added in
        // future sqlparser versions, we'll get a compilation error and can handle them
        let Select {
            // Supported clauses
            distinct: _,
            from: _,
            group_by: _,
            having: _,
            named_window: _,
            projection: _,
            qualify: _,
            selection: _,

            // Metadata/token fields (can ignore)
            flavor: _,
            select_token: _,
            top_before_distinct: _,
            window_before_qualify: _,

            // Unsupported clauses
            ref cluster_by,
            ref connect_by,
            ref distribute_by,
            ref exclude,
            ref into,
            ref lateral_views,
            ref prewhere,
            ref sort_by,
            ref top,
            ref value_table_mode,
        } = *select_stmt;

        // Raise specific error messages for unsupported attributes
        polars_ensure!(cluster_by.is_empty(), SQLInterface: "`CLUSTER BY` clause is not supported");
        polars_ensure!(connect_by.is_none(), SQLInterface: "`CONNECT BY` clause is not supported");
        polars_ensure!(distribute_by.is_empty(), SQLInterface: "`DISTRIBUTE BY` clause is not supported");
        polars_ensure!(exclude.is_none(), SQLInterface: "`EXCLUDE` clause is not supported");
        polars_ensure!(into.is_none(), SQLInterface: "`SELECT INTO` clause is not supported");
        polars_ensure!(lateral_views.is_empty(), SQLInterface: "`LATERAL VIEW` clause is not supported");
        polars_ensure!(prewhere.is_none(), SQLInterface: "`PREWHERE` clause is not supported");
        polars_ensure!(sort_by.is_empty(), SQLInterface: "`SORT BY` clause is not supported; use `ORDER BY` instead");
        polars_ensure!(top.is_none(), SQLInterface: "`TOP` clause is not supported; use `LIMIT` instead");
        polars_ensure!(value_table_mode.is_none(), SQLInterface: "`SELECT AS VALUE/STRUCT` is not supported");

        Ok(())
    }

    /// Check that the QUERY only contains supported clauses.
    fn validate_query(&self, query: &Query) -> PolarsResult<()> {
        // As with "Select" validation (above) destructure "Query" exhaustively
        let Query {
            // Supported clauses
            with: _,
            body: _,
            order_by: _,
            limit_clause: _,
            fetch,

            // Unsupported clauses
            for_clause,
            format_clause,
            locks,
            pipe_operators,
            settings,
        } = query;

        // Raise specific error messages for unsupported attributes
        polars_ensure!(for_clause.is_none(), SQLInterface: "`FOR` clause is not supported");
        polars_ensure!(format_clause.is_none(), SQLInterface: "`FORMAT` clause is not supported");
        polars_ensure!(locks.is_empty(), SQLInterface: "`FOR UPDATE/SHARE` locking clause is not supported");
        polars_ensure!(pipe_operators.is_empty(), SQLInterface: "pipe operators are not supported");
        polars_ensure!(settings.is_none(), SQLInterface: "`SETTINGS` clause is not supported");

        // Validate FETCH clause options (if present)
        if let Some(Fetch {
            quantity: _, // supported
            percent,
            with_ties,
        }) = fetch
        {
            polars_ensure!(!percent, SQLInterface: "`FETCH` with `PERCENT` is not supported");
            polars_ensure!(!with_ties, SQLInterface: "`FETCH` with `WITH TIES` is not supported");
        }
        Ok(())
    }

    /// Execute the 'SELECT' part of the query.
    fn execute_select(&mut self, select_stmt: &Select, query: &Query) -> PolarsResult<LazyFrame> {
        // Check that the statement doesn't contain unsupported SELECT clauses
        self.validate_select(select_stmt)?;

        // Parse named windows first, as they may be referenced in the SELECT clause
        self.register_named_windows(&select_stmt.named_window)?;

        // Get `FROM` table/data
        let (mut lf, base_table_name) = if select_stmt.from.is_empty() {
            (DataFrame::empty().lazy(), None)
        } else {
            // Note: implicit joins need more work to support properly,
            // explicit joins are preferred for now (ref: #16662)
            let from = select_stmt.clone().from;
            if from.len() > 1 {
                polars_bail!(SQLInterface: "multiple tables in FROM clause are not currently supported (found {}); use explicit JOIN syntax instead", from.len())
            }
            let tbl_expr = from.first().unwrap();
            let lf = self.execute_from_statement(tbl_expr)?;
            let base_name = get_table_name(&tbl_expr.relation);
            (lf, base_name)
        };

        // Check for ambiguous column references in SELECT and WHERE (if there were joins)
        if let Some(ref base_name) = base_table_name {
            if !self.joined_aliases.is_empty() {
                // Extract USING columns from joins (these are coalesced and not ambiguous)
                let using_cols: PlHashSet<String> = select_stmt
                    .from
                    .first()
                    .into_iter()
                    .flat_map(|t| t.joins.iter())
                    .filter_map(|join| get_using_cols(&join.join_operator))
                    .flatten()
                    .collect();

                // Check SELECT and WHERE expressions for ambiguous column references
                let check_expr = |e| {
                    check_for_ambiguous_column_refs(e, &self.joined_aliases, base_name, &using_cols)
                };
                for item in &select_stmt.projection {
                    match item {
                        SelectItem::UnnamedExpr(e) | SelectItem::ExprWithAlias { expr: e, .. } => {
                            check_expr(e)?
                        },
                        _ => {},
                    }
                }
                if let Some(ref where_expr) = select_stmt.selection {
                    check_expr(where_expr)?;
                }
            }
        }

        // Apply `WHERE` constraint
        let mut schema = self.get_frame_schema(&mut lf)?;
        lf = self.process_where(lf, &select_stmt.selection, false, Some(schema.clone()))?;

        // Determine projections
        let mut select_modifiers = SelectModifiers {
            ilike: None,
            exclude: PlHashSet::new(),
            rename: PlHashMap::new(),
            replace: vec![],
        };

        // Collect window function cols if QUALIFY is present (we check at the
        // SQL level because empty OVER() clauses don't create Expr::Over)
        let window_fn_columns = if select_stmt.qualify.is_some() {
            select_stmt
                .projection
                .iter()
                .filter_map(|item| match item {
                    SelectItem::ExprWithAlias { expr, alias }
                        if expr_has_window_functions(expr) =>
                    {
                        Some(alias.value.clone())
                    },
                    _ => None,
                })
                .collect::<PlHashSet<_>>()
        } else {
            PlHashSet::new()
        };

        let mut projections =
            self.column_projections(select_stmt, &schema, &mut select_modifiers)?;

        // Apply `UNNEST` expressions
        let mut explode_names = Vec::new();
        let mut explode_exprs = Vec::new();
        let mut explode_lookup = PlHashMap::new();

        for expr in &projections {
            for e in expr {
                if let Expr::Explode { input, .. } = e {
                    match input.as_ref() {
                        Expr::Column(name) => explode_names.push(name.clone()),
                        other_expr => {
                            // Note: skip aggregate expressions; those are handled in the GROUP BY phase
                            if !has_expr(other_expr, |e| matches!(e, Expr::Agg(_) | Expr::Len)) {
                                let temp_name =
                                    format_pl_smallstr!("__POLARS_UNNEST_{}", explode_exprs.len());
                                explode_exprs.push(other_expr.clone().alias(temp_name.as_str()));
                                explode_lookup.insert(other_expr.clone(), temp_name.clone());
                                explode_names.push(temp_name);
                            }
                        },
                    }
                }
            }
        }
        if !explode_names.is_empty() {
            if !explode_exprs.is_empty() {
                lf = lf.with_columns(explode_exprs);
            }
            lf = lf.explode(
                Selector::ByName {
                    names: Arc::from(explode_names),
                    strict: true,
                },
                ExplodeOptions {
                    empty_as_null: true,
                    keep_nulls: true,
                },
            );
            projections = projections
                .into_iter()
                .map(|p| {
                    // Update "projections" with column refs to the now-exploded expressions
                    p.map_expr(|e| match e {
                        Expr::Explode { input, .. } => explode_lookup
                            .get(input.as_ref())
                            .map(|name| Expr::Column(name.clone()))
                            .unwrap_or_else(|| input.as_ref().clone()),
                        _ => e,
                    })
                })
                .collect();

            schema = self.get_frame_schema(&mut lf)?;
        }

        // Check for "GROUP BY ..." (after determining projections)
        let mut group_by_keys: Vec<Expr> = Vec::new();
        match &select_stmt.group_by {
            // Standard "GROUP BY x, y, z" syntax (also recognising ordinal values)
            GroupByExpr::Expressions(group_by_exprs, modifiers) => {
                if !modifiers.is_empty() {
                    polars_bail!(SQLInterface: "GROUP BY does not support CUBE, ROLLUP, or TOTALS modifiers")
                }
                // Translate the group expressions, resolving ordinal values and SELECT aliases
                group_by_keys = group_by_exprs
                    .iter()
                    .map(|e| match e {
                        SQLExpr::Identifier(ident) => {
                            resolve_select_alias(&ident.value, &projections, &schema).map_or_else(
                                || {
                                    self.expr_or_ordinal(
                                        e,
                                        &projections,
                                        None,
                                        Some(&schema),
                                        "GROUP BY",
                                    )
                                },
                                Ok,
                            )
                        },
                        _ => self.expr_or_ordinal(e, &projections, None, Some(&schema), "GROUP BY"),
                    })
                    .collect::<PolarsResult<_>>()?
            },
            // "GROUP BY ALL" syntax; automatically adds expressions that do not contain
            // nested agg/window funcs to the group key (also ignores literals).
            GroupByExpr::All(modifiers) => {
                if !modifiers.is_empty() {
                    polars_bail!(SQLInterface: "GROUP BY does not support CUBE, ROLLUP, or TOTALS modifiers")
                }
                projections.iter().for_each(|expr| match expr {
                    // immediately match the most common cases (col|agg|len|lit, optionally aliased).
                    Expr::Agg(_) | Expr::Len | Expr::Literal(_) => (),
                    Expr::Column(_) => group_by_keys.push(expr.clone()),
                    Expr::Alias(e, _)
                        if matches!(&**e, Expr::Agg(_) | Expr::Len | Expr::Literal(_)) => {},
                    Expr::Alias(e, _) if matches!(&**e, Expr::Column(_)) => {
                        if let Expr::Column(name) = &**e {
                            group_by_keys.push(col(name.clone()));
                        }
                    },
                    _ => {
                        // If not quick-matched, add if no nested agg/window expressions
                        if !has_expr(expr, |e| {
                            matches!(e, Expr::Agg(_))
                                || matches!(e, Expr::Len)
                                || matches!(e, Expr::Over { .. })
                                || {
                                    #[cfg(feature = "dynamic_group_by")]
                                    {
                                        matches!(e, Expr::Rolling { .. })
                                    }
                                    #[cfg(not(feature = "dynamic_group_by"))]
                                    {
                                        false
                                    }
                                }
                        }) {
                            group_by_keys.push(expr.clone())
                        }
                    },
                });
            },
        };

        lf = if group_by_keys.is_empty() {
            // The 'having' clause is only valid inside 'group by'
            if select_stmt.having.is_some() {
                polars_bail!(SQLSyntax: "HAVING clause not valid outside of GROUP BY; found:\n{:?}", select_stmt.having);
            };

            // Final/selected cols, accounting for 'SELECT *' modifiers
            let mut retained_cols = Vec::with_capacity(projections.len());
            let mut retained_names = Vec::with_capacity(projections.len());
            let have_order_by = query.order_by.is_some();

            // Initialize containing InheritsContext to handle empty projection case.
            let mut projection_heights = ExprSqlProjectionHeightBehavior::InheritsContext;

            // Note: if there is an 'order by' then we project everything (original cols
            // and new projections) and *then* select the final cols; the retained cols
            // are used to ensure a correct final projection. If there's no 'order by',
            // clause then we can project the final column *expressions* directly.
            for p in projections.iter() {
                let name = p.to_field(schema.deref())?.name.to_string();
                if select_modifiers.matches_ilike(&name)
                    && !select_modifiers.exclude.contains(&name)
                {
                    projection_heights |= ExprSqlProjectionHeightBehavior::identify_from_expr(p);

                    retained_cols.push(if have_order_by {
                        col(name.as_str())
                    } else {
                        p.clone()
                    });
                    retained_names.push(col(name));
                }
            }

            // Apply the remaining modifiers and establish the final projection
            if have_order_by {
                // We can safely use `with_columns()` and avoid a join if:
                // * There is already a projection that projects to the table height.
                // * All projection heights inherit from context (e.g. all scalar literals that
                //   are to be broadcasted to table height).
                if projection_heights.contains(ExprSqlProjectionHeightBehavior::MaintainsColumn)
                    || projection_heights == ExprSqlProjectionHeightBehavior::InheritsContext
                {
                    lf = lf.with_columns(projections);
                } else {
                    // We hit this branch if the output height is not guaranteed to match the table
                    // height. E.g.:
                    //
                    // * SELECT COUNT(*) FROM df ORDER BY sort_key;
                    //
                    // For these cases we truncate / extend the sorting columns with NULLs to match
                    // the output height. We do this by projecting independently and then joining
                    // back the original frame on the row index.
                    const NAME: PlSmallStr = PlSmallStr::from_static("__PL_INDEX");
                    lf = lf
                        .clone()
                        .select(projections)
                        .with_row_index(NAME, None)
                        .join(
                            lf.with_row_index(NAME, None),
                            [col(NAME)],
                            [col(NAME)],
                            JoinArgs {
                                how: JoinType::Left,
                                validation: Default::default(),
                                suffix: None,
                                slice: None,
                                nulls_equal: false,
                                coalesce: Default::default(),
                                maintain_order: MaintainOrderJoin::Left,
                                build_side: None,
                            },
                        );
                }
            }
            if !select_modifiers.replace.is_empty() {
                lf = lf.with_columns(&select_modifiers.replace);
            }
            if !select_modifiers.rename.is_empty() {
                lf = lf.with_columns(select_modifiers.renamed_cols());
            }
            lf = self.process_order_by(lf, &query.order_by, Some(&retained_cols))?;

            // Note: If `have_order_by`, with_columns is already done above.
            if projection_heights == ExprSqlProjectionHeightBehavior::InheritsContext
                && !have_order_by
            {
                // All projections need to be broadcasted to table height, so evaluate in `with_columns()`
                lf = lf.with_columns(retained_cols).select(retained_names);
            } else {
                lf = lf.select(retained_cols);
            }
            if !select_modifiers.rename.is_empty() {
                lf = lf.rename(
                    select_modifiers.rename.keys(),
                    select_modifiers.rename.values(),
                    true,
                );
            };
            lf
        } else {
            let having = select_stmt
                .having
                .as_ref()
                .map(|expr| parse_sql_expr(expr, self, Some(&schema)))
                .transpose()?;
            lf = self.process_group_by(lf, &group_by_keys, &projections, having)?;
            lf = self.process_order_by(lf, &query.order_by, None)?;

            // Drop any extra columns (eg: added to maintain ORDER BY access to original cols)
            let output_cols: Vec<_> = projections
                .iter()
                .map(|p| p.to_field(&schema))
                .collect::<PolarsResult<Vec<_>>>()?
                .into_iter()
                .map(|f| col(f.name))
                .collect();

            lf.select(&output_cols)
        };

        // Apply optional QUALIFY clause (filters on window functions).
        lf = self.process_qualify(lf, &select_stmt.qualify, &window_fn_columns)?;

        // Apply optional DISTINCT clause.
        lf = match &select_stmt.distinct {
            Some(Distinct::Distinct) => lf.unique_stable(None, UniqueKeepStrategy::Any),
            Some(Distinct::On(exprs)) => {
                // TODO: support exprs in `unique` see https://github.com/pola-rs/polars/issues/5760
                let schema = Some(self.get_frame_schema(&mut lf)?);
                let cols = exprs
                    .iter()
                    .map(|e| {
                        let expr = parse_sql_expr(e, self, schema.as_deref())?;
                        if let Expr::Column(name) = expr {
                            Ok(name)
                        } else {
                            Err(polars_err!(SQLSyntax:"DISTINCT ON only supports column names"))
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                // DISTINCT ON has to apply the ORDER BY before the operation.
                lf = self.process_order_by(lf, &query.order_by, None)?;
                return Ok(lf.unique_stable(
                    Some(Selector::ByName {
                        names: cols.into(),
                        strict: true,
                    }),
                    UniqueKeepStrategy::First,
                ));
            },
            None => lf,
        };
        Ok(lf)
    }

    fn column_projections(
        &mut self,
        select_stmt: &Select,
        schema: &SchemaRef,
        select_modifiers: &mut SelectModifiers,
    ) -> PolarsResult<Vec<Expr>> {
        if select_stmt.projection.is_empty()
            && select_stmt.flavor == SelectFlavor::FromFirstNoSelect
        {
            // eg: bare "FROM tbl" is equivalent to "SELECT * FROM tbl".
            return Ok(schema.iter_names().map(|name| col(name.clone())).collect());
        }
        let mut items: Vec<ProjectionItem> = Vec::with_capacity(select_stmt.projection.len());
        let mut has_qualified_wildcard = false;

        for select_item in &select_stmt.projection {
            match select_item {
                SelectItem::UnnamedExpr(expr) => {
                    items.push(ProjectionItem::Exprs(vec![parse_sql_expr(
                        expr,
                        self,
                        Some(schema),
                    )?]));
                },
                SelectItem::ExprWithAlias { expr, alias } => {
                    let expr = parse_sql_expr(expr, self, Some(schema))?;
                    items.push(ProjectionItem::Exprs(vec![
                        expr.alias(PlSmallStr::from_str(alias.value.as_str())),
                    ]));
                },
                SelectItem::QualifiedWildcard(kind, wildcard_options) => match kind {
                    SelectItemQualifiedWildcardKind::ObjectName(obj_name) => {
                        let tbl_name = obj_name
                            .0
                            .last()
                            .and_then(|p| p.as_ident())
                            .map(|i| PlSmallStr::from_str(&i.value))
                            .unwrap_or_default();
                        let exprs = self.process_qualified_wildcard(
                            obj_name,
                            wildcard_options,
                            select_modifiers,
                            Some(schema),
                        )?;
                        items.push(ProjectionItem::QualifiedExprs(tbl_name, exprs));
                        has_qualified_wildcard = true;
                    },
                    SelectItemQualifiedWildcardKind::Expr(_) => {
                        polars_bail!(SQLSyntax: "qualified wildcard on expressions not yet supported: {:?}", select_item)
                    },
                },
                SelectItem::Wildcard(wildcard_options) => {
                    let cols = schema.iter_names().map(|name| col(name.clone())).collect();
                    items.push(ProjectionItem::Exprs(
                        self.process_wildcard_additional_options(
                            cols,
                            wildcard_options,
                            select_modifiers,
                            Some(schema),
                        )?,
                    ));
                },
            }
        }

        // Disambiguate qualified wildcards (if any) and flatten expressions
        let exprs = if has_qualified_wildcard {
            disambiguate_projection_cols(items, schema)?
        } else {
            items
                .into_iter()
                .flat_map(|item| match item {
                    ProjectionItem::Exprs(exprs) | ProjectionItem::QualifiedExprs(_, exprs) => {
                        exprs
                    },
                })
                .collect()
        };
        let flattened_exprs = exprs
            .into_iter()
            .flat_map(|expr| expand_exprs(expr, schema))
            .collect();

        Ok(flattened_exprs)
    }

    fn process_where(
        &mut self,
        mut lf: LazyFrame,
        expr: &Option<SQLExpr>,
        invert_filter: bool,
        schema: Option<SchemaRef>,
    ) -> PolarsResult<LazyFrame> {
        if let Some(expr) = expr {
            let schema = match schema {
                None => self.get_frame_schema(&mut lf)?,
                Some(s) => s,
            };

            // shortcut filter evaluation if given expression is just TRUE or FALSE
            let (all_true, all_false) = match expr {
                SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Boolean(b),
                    ..
                }) => (*b, !*b),
                SQLExpr::BinaryOp { left, op, right } => match (&**left, &**right, op) {
                    (SQLExpr::Value(a), SQLExpr::Value(b), BinaryOperator::Eq) => {
                        (a.value == b.value, a.value != b.value)
                    },
                    (SQLExpr::Value(a), SQLExpr::Value(b), BinaryOperator::NotEq) => {
                        (a.value != b.value, a.value == b.value)
                    },
                    _ => (false, false),
                },
                _ => (false, false),
            };
            if (all_true && !invert_filter) || (all_false && invert_filter) {
                return Ok(lf);
            } else if (all_false && !invert_filter) || (all_true && invert_filter) {
                return Ok(lf.clear());
            }

            // ...otherwise parse and apply the filter as normal
            let mut filter_expression = parse_sql_expr(expr, self, Some(schema).as_deref())?;
            if filter_expression.clone().meta().has_multiple_outputs() {
                filter_expression = all_horizontal([filter_expression])?;
            }
            lf = self.process_subqueries(lf, vec![&mut filter_expression]);
            lf = if invert_filter {
                lf.remove(filter_expression)
            } else {
                lf.filter(filter_expression)
            };
        }
        Ok(lf)
    }

    pub(super) fn process_join(
        &mut self,
        tbl_left: &TableInfo,
        tbl_right: &TableInfo,
        constraint: &JoinConstraint,
        join_type: JoinType,
    ) -> PolarsResult<LazyFrame> {
        let (left_on, right_on) = process_join_constraint(constraint, tbl_left, tbl_right, self)?;
        let coalesce_type = match constraint {
            // "NATURAL" joins should coalesce; otherwise we disambiguate
            JoinConstraint::Natural => JoinCoalesce::CoalesceColumns,
            _ => JoinCoalesce::KeepColumns,
        };
        let joined = tbl_left
            .frame
            .clone()
            .join_builder()
            .with(tbl_right.frame.clone())
            .left_on(left_on)
            .right_on(right_on)
            .how(join_type)
            .suffix(format!(":{}", tbl_right.name))
            .coalesce(coalesce_type)
            .finish();

        Ok(joined)
    }

    fn process_qualify(
        &mut self,
        mut lf: LazyFrame,
        qualify_expr: &Option<SQLExpr>,
        window_fn_columns: &PlHashSet<String>,
    ) -> PolarsResult<LazyFrame> {
        if let Some(expr) = qualify_expr {
            // Check the QUALIFY expression to identify window functions
            // and collect column refs (for looking up aliases from SELECT)
            let (has_window_fns, column_refs) = QualifyExpression::analyze(expr);
            let references_window_alias = column_refs.iter().any(|c| window_fn_columns.contains(c));
            if !has_window_fns && !references_window_alias {
                polars_bail!(
                    SQLSyntax:
                    "QUALIFY clause must reference window functions either explicitly or via SELECT aliases"
                );
            }
            let schema = self.get_frame_schema(&mut lf)?;
            let mut filter_expression = parse_sql_expr(expr, self, Some(&schema))?;
            if filter_expression.clone().meta().has_multiple_outputs() {
                filter_expression = all_horizontal([filter_expression])?;
            }
            lf = self.process_subqueries(lf, vec![&mut filter_expression]);
            lf = lf.filter(filter_expression);
        }
        Ok(lf)
    }

    fn process_subqueries(&self, lf: LazyFrame, exprs: Vec<&mut Expr>) -> LazyFrame {
        let mut subplans = vec![];

        for e in exprs {
            *e = e.clone().map_expr(|e| {
                if let Expr::SubPlan(lp, names) = e {
                    assert_eq!(
                        names.len(),
                        1,
                        "multiple columns in subqueries not yet supported"
                    );

                    let select_expr = names[0].1.clone();
                    let cb =
                        PlanCallback::new(move |(plans, schemas): (Vec<DslPlan>, Vec<SchemaRef>)| {
                            let schema = &schemas[0];
                            polars_ensure!(schema.len() == 1,  SQLSyntax: "SQL subquery returns more than one column");
                            Ok(LazyFrame::from(plans.into_iter().next().unwrap()).select([select_expr.clone()]).logical_plan)
                        });
                    subplans.push(LazyFrame::from((**lp).clone()).pipe_with_schema(cb));
                    Expr::Column(names[0].0.clone()).first()
                } else {
                    e
                }
            });
        }

        if subplans.is_empty() {
            lf
        } else {
            subplans.insert(0, lf);
            concat_lf_horizontal(
                subplans,
                HConcatOptions {
                    broadcast_unit_length: true,
                    ..Default::default()
                },
            )
            .unwrap()
        }
    }

    fn execute_create_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        if let Statement::CreateTable(CreateTable {
            if_not_exists,
            name,
            query,
            columns,
            like,
            ..
        }) = stmt
        {
            let tbl_name = name.0.first().unwrap().as_ident().unwrap().value.as_str();
            if *if_not_exists && self.table_map.read().unwrap().contains_key(tbl_name) {
                polars_bail!(SQLInterface: "relation '{}' already exists", tbl_name);
            }
            let lf = match (query, columns.is_empty(), like) {
                (Some(query), true, None) => {
                    // ----------------------------------------------------
                    // CREATE TABLE [IF NOT EXISTS] <name> AS <query>
                    // ----------------------------------------------------
                    self.execute_query(query)?
                },
                (None, false, None) => {
                    // ----------------------------------------------------
                    // CREATE TABLE [IF NOT EXISTS] <name> (<coldef>, ...)
                    // ----------------------------------------------------
                    let mut schema = Schema::with_capacity(columns.len());
                    for col in columns {
                        let col_name = col.name.value.as_str();
                        let dtype = map_sql_dtype_to_polars(&col.data_type)?;
                        schema.insert_at_index(schema.len(), col_name.into(), dtype)?;
                    }
                    DataFrame::empty_with_schema(&schema).lazy()
                },
                (None, true, Some(like_kind)) => {
                    // ----------------------------------------------------
                    // CREATE TABLE [IF NOT EXISTS] <name> LIKE <table>
                    // ----------------------------------------------------
                    let like_name = match like_kind {
                        CreateTableLikeKind::Plain(like)
                        | CreateTableLikeKind::Parenthesized(like) => &like.name,
                    };
                    let like_table = like_name
                        .0
                        .first()
                        .unwrap()
                        .as_ident()
                        .unwrap()
                        .value
                        .as_str();
                    if let Some(table) = self.table_map.read().unwrap().get(like_table).cloned() {
                        table.clear()
                    } else {
                        polars_bail!(SQLInterface: "table given in LIKE does not exist: {}", like_table)
                    }
                },
                // No valid options provided
                (None, true, None) => {
                    polars_bail!(SQLInterface: "CREATE TABLE expected a query, column definitions, or LIKE clause")
                },
                // Mutually exclusive options
                _ => {
                    polars_bail!(
                        SQLInterface: "CREATE TABLE received mutually exclusive options:\nquery = {:?}\ncolumns = {:?}\nlike = {:?}",
                        query,
                        columns,
                        like,
                    )
                },
            };
            self.register(tbl_name, lf);

            let df_created = df! { "Response" => [format!("CREATE TABLE {}", name.0.first().unwrap().as_ident().unwrap().value)] };
            Ok(df_created.unwrap().lazy())
        } else {
            unreachable!()
        }
    }

    fn get_table(&mut self, relation: &TableFactor) -> PolarsResult<(String, LazyFrame)> {
        match relation {
            TableFactor::Table {
                name, alias, args, ..
            } => {
                if let Some(args) = args {
                    return self.execute_table_function(name, alias, &args.args);
                }
                let tbl_name = name.0.first().unwrap().as_ident().unwrap().value.as_str();
                if let Some(lf) = self.get_table_from_current_scope(tbl_name) {
                    match alias {
                        Some(alias) => {
                            self.table_aliases
                                .insert(alias.name.value.clone(), tbl_name.to_string());
                            Ok((alias.name.value.clone(), lf))
                        },
                        None => Ok((tbl_name.to_string(), lf)),
                    }
                } else {
                    polars_bail!(SQLInterface: "relation '{}' was not found", tbl_name);
                }
            },
            TableFactor::Derived {
                lateral,
                subquery,
                alias,
            } => {
                polars_ensure!(!(*lateral), SQLInterface: "LATERAL not supported");
                if let Some(alias) = alias {
                    let mut lf = self.execute_query_no_ctes(subquery)?;
                    lf = self.rename_columns_from_table_alias(lf, alias)?;
                    self.table_map
                        .write()
                        .unwrap()
                        .insert(alias.name.value.clone(), lf.clone());
                    Ok((alias.name.value.clone(), lf))
                } else {
                    let lf = self.execute_query_no_ctes(subquery)?;
                    Ok(("".to_string(), lf))
                }
            },
            TableFactor::UNNEST {
                alias,
                array_exprs,
                with_offset,
                with_offset_alias: _,
                ..
            } => {
                if let Some(alias) = alias {
                    let column_names: Vec<Option<PlSmallStr>> = alias
                        .columns
                        .iter()
                        .map(|c| {
                            if c.name.value.is_empty() {
                                None
                            } else {
                                Some(PlSmallStr::from_str(c.name.value.as_str()))
                            }
                        })
                        .collect();

                    let column_values: Vec<Series> = array_exprs
                        .iter()
                        .map(|arr| parse_sql_array(arr, self))
                        .collect::<Result<_, _>>()?;

                    polars_ensure!(!column_names.is_empty(),
                        SQLSyntax:
                        "UNNEST table alias must also declare column names, eg: {} (a,b,c)", alias.name.to_string()
                    );
                    if column_names.len() != column_values.len() {
                        let plural = if column_values.len() > 1 { "s" } else { "" };
                        polars_bail!(
                            SQLSyntax:
                            "UNNEST table alias requires {} column name{}, found {}", column_values.len(), plural, column_names.len()
                        );
                    }
                    let column_series: Vec<Column> = column_values
                        .into_iter()
                        .zip(column_names)
                        .map(|(s, name)| {
                            if let Some(name) = name {
                                s.with_name(name)
                            } else {
                                s
                            }
                        })
                        .map(Column::from)
                        .collect();

                    let lf = DataFrame::new_infer_height(column_series)?.lazy();

                    if *with_offset {
                        // TODO: support 'WITH ORDINALITY|OFFSET' modifier.
                        polars_bail!(SQLInterface: "UNNEST tables do not (yet) support WITH ORDINALITY|OFFSET");
                    }
                    let table_name = alias.name.value.clone();
                    self.table_map
                        .write()
                        .unwrap()
                        .insert(table_name.clone(), lf.clone());
                    Ok((table_name, lf))
                } else {
                    polars_bail!(SQLSyntax: "UNNEST table must have an alias");
                }
            },
            TableFactor::NestedJoin {
                table_with_joins,
                alias,
            } => {
                let lf = self.execute_from_statement(table_with_joins)?;
                match alias {
                    Some(a) => Ok((a.name.value.clone(), lf)),
                    None => Ok(("".to_string(), lf)),
                }
            },
            // Support bare table, optionally with an alias, for now
            _ => polars_bail!(SQLInterface: "not yet implemented: {}", relation),
        }
    }

    fn execute_table_function(
        &mut self,
        name: &ObjectName,
        alias: &Option<TableAlias>,
        args: &[FunctionArg],
    ) -> PolarsResult<(String, LazyFrame)> {
        let tbl_fn = name.0.first().unwrap().as_ident().unwrap().value.as_str();
        let read_fn = tbl_fn.parse::<PolarsTableFunctions>()?;
        let (tbl_name, lf) = read_fn.execute(args)?;
        #[allow(clippy::useless_asref)]
        let tbl_name = alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or_else(|| tbl_name.to_string());

        self.table_map
            .write()
            .unwrap()
            .insert(tbl_name.clone(), lf.clone());
        Ok((tbl_name, lf))
    }

    fn process_order_by(
        &mut self,
        mut lf: LazyFrame,
        order_by: &Option<OrderBy>,
        selected: Option<&[Expr]>,
    ) -> PolarsResult<LazyFrame> {
        if order_by.as_ref().is_none_or(|ob| match &ob.kind {
            OrderByKind::Expressions(exprs) => exprs.is_empty(),
            OrderByKind::All(_) => false,
        }) {
            return Ok(lf);
        }
        let schema = self.get_frame_schema(&mut lf)?;
        let columns_iter = schema.iter_names().map(|e| col(e.clone()));
        let (order_by, order_by_all, n_order_cols) = match &order_by.as_ref().unwrap().kind {
            OrderByKind::Expressions(exprs) => {
                // TODO: will look at making an upstream PR that allows us to more easily
                //  create a GenericDialect variant supporting "OrderByKind::All" instead
                if exprs.len() == 1
                    && matches!(&exprs[0].expr, SQLExpr::Identifier(ident)
                        if ident.value.to_uppercase() == "ALL"
                        && !schema.iter_names().any(|name| name.to_uppercase() == "ALL"))
                {
                    // Treat as ORDER BY ALL
                    let n_cols = if let Some(selected) = selected {
                        selected.len()
                    } else {
                        schema.len()
                    };
                    (vec![], Some(&exprs[0].options), n_cols)
                } else {
                    (exprs.clone(), None, exprs.len())
                }
            },
            OrderByKind::All(opts) => {
                let n_cols = if let Some(selected) = selected {
                    selected.len()
                } else {
                    schema.len()
                };
                (vec![], Some(opts), n_cols)
            },
        };
        let mut descending = Vec::with_capacity(n_order_cols);
        let mut nulls_last = Vec::with_capacity(n_order_cols);
        let mut by: Vec<Expr> = Vec::with_capacity(n_order_cols);

        if let Some(opts) = order_by_all {
            if let Some(selected) = selected {
                by.extend(selected.iter().cloned());
            } else {
                by.extend(columns_iter);
            };
            let desc_order = !opts.asc.unwrap_or(true);
            nulls_last.resize(by.len(), !opts.nulls_first.unwrap_or(desc_order));
            descending.resize(by.len(), desc_order);
        } else {
            let columns = &columns_iter.collect::<Vec<_>>();
            for ob in order_by {
                // note: if not specified 'NULLS FIRST' is default for DESC, 'NULLS LAST' otherwise
                // https://www.postgresql.org/docs/current/queries-order.html
                let desc_order = !ob.options.asc.unwrap_or(true);
                nulls_last.push(!ob.options.nulls_first.unwrap_or(desc_order));
                descending.push(desc_order);

                // translate order expression, allowing ordinal values
                by.push(self.expr_or_ordinal(
                    &ob.expr,
                    columns,
                    selected,
                    Some(&schema),
                    "ORDER BY",
                )?)
            }
        }
        Ok(lf.sort_by_exprs(
            &by,
            SortMultipleOptions::default()
                .with_order_descending_multi(descending)
                .with_nulls_last_multi(nulls_last),
        ))
    }

    fn process_group_by(
        &mut self,
        mut lf: LazyFrame,
        group_by_keys: &[Expr],
        projections: &[Expr],
        having: Option<Expr>,
    ) -> PolarsResult<LazyFrame> {
        let schema_before = self.get_frame_schema(&mut lf)?;
        let group_by_keys_schema =
            expressions_to_schema(group_by_keys, &schema_before, |duplicate_name: &str| {
                format!("group_by keys contained duplicate output name '{duplicate_name}'")
            })?;

        // Note: remove the `group_by` keys as Polars adds those implicitly.
        let mut aliased_aggregations: PlHashMap<PlSmallStr, PlSmallStr> = PlHashMap::new();
        let mut aggregation_projection = Vec::with_capacity(projections.len());
        let mut projection_overrides = PlHashMap::with_capacity(projections.len());
        let mut projection_aliases = PlHashSet::new();
        let mut group_key_aliases = PlHashSet::new();

        // Pre-compute group key data (stripped expression + output name) to avoid repeated work.
        // We check both expression AND output name match to avoid cross-aliasing issues.
        let group_key_data: Vec<_> = group_by_keys
            .iter()
            .map(|gk| {
                (
                    strip_outer_alias(gk),
                    gk.to_field(&schema_before).ok().map(|f| f.name),
                )
            })
            .collect();

        let projection_matches_group_key: Vec<bool> = projections
            .iter()
            .map(|p| {
                let p_stripped = strip_outer_alias(p);
                let p_name = p.to_field(&schema_before).ok().map(|f| f.name);
                group_key_data
                    .iter()
                    .any(|(gk_stripped, gk_name)| *gk_stripped == p_stripped && *gk_name == p_name)
            })
            .collect();

        for (e, &matches_group_key) in projections.iter().zip(&projection_matches_group_key) {
            // `Len` represents COUNT(*) so we treat as an aggregation here.
            let is_non_group_key_expr = !matches_group_key
                && has_expr(e, |e| {
                    match e {
                        Expr::Agg(_) | Expr::Len | Expr::Over { .. } => true,
                        #[cfg(feature = "dynamic_group_by")]
                        Expr::Rolling { .. } => true,
                        Expr::Function { function: func, .. }
                            if !matches!(func, FunctionExpr::StructExpr(_)) =>
                        {
                            // If it's a function call containing a column NOT in the group by keys,
                            // we treat it as an aggregation.
                            has_expr(e, |e| match e {
                                Expr::Column(name) => !group_by_keys_schema.contains(name),
                                _ => false,
                            })
                        },
                        _ => false,
                    }
                });

            // Note: if simple aliased expression we defer aliasing until after the group_by.
            // Use `e_inner` to track the potentially unwrapped expression for field lookup.
            let mut e_inner = e;
            if let Expr::Alias(expr, alias) = e {
                if e.clone().meta().is_simple_projection(Some(&schema_before)) {
                    group_key_aliases.insert(alias.as_ref());
                    e_inner = expr
                } else if let Expr::Function {
                    function: FunctionExpr::StructExpr(StructFunction::FieldByName(name)),
                    ..
                } = expr.deref()
                {
                    projection_overrides
                        .insert(alias.as_ref(), col(name.clone()).alias(alias.clone()));
                } else if !is_non_group_key_expr && !group_by_keys_schema.contains(alias) {
                    projection_aliases.insert(alias.as_ref());
                }
            }
            let field = e_inner.to_field(&schema_before)?;
            if is_non_group_key_expr {
                let mut e = e.clone();
                if let Expr::Agg(AggExpr::Implode(expr)) = &e {
                    e = (**expr).clone();
                } else if let Expr::Alias(expr, name) = &e {
                    if let Expr::Agg(AggExpr::Implode(expr)) = expr.as_ref() {
                        e = (**expr).clone().alias(name.clone());
                    }
                }
                // If aggregation colname conflicts with a group key,
                // alias it to avoid duplicate/mis-tracked columns
                if group_by_keys_schema.get(&field.name).is_some() {
                    let alias_name = format!("__POLARS_AGG_{}", field.name);
                    e = e.alias(alias_name.as_str());
                    aliased_aggregations.insert(field.name.clone(), alias_name.as_str().into());
                }
                aggregation_projection.push(e);
            } else if !matches_group_key {
                // Non-aggregated columns must be part of the GROUP BY clause
                if let Expr::Column(_)
                | Expr::Function {
                    function: FunctionExpr::StructExpr(StructFunction::FieldByName(_)),
                    ..
                } = e_inner
                {
                    if !group_by_keys_schema.contains(&field.name) {
                        polars_bail!(SQLSyntax: "'{}' should participate in the GROUP BY clause or an aggregate function", &field.name);
                    }
                }
            }
        }

        // Process HAVING clause: identify aggregate expressions, reusing those already
        // in projections, or compute as temporary columns and then post-filter/discard
        let having_filter = if let Some(having_expr) = having {
            let mut agg_to_name: Vec<(Expr, PlSmallStr)> = aggregation_projection
                .iter()
                .filter_map(|p| match p {
                    Expr::Alias(inner, name) if matches!(**inner, Expr::Agg(_) | Expr::Len) => {
                        Some((inner.as_ref().clone(), name.clone()))
                    },
                    e @ (Expr::Agg(_) | Expr::Len) => Some((
                        e.clone(),
                        e.to_field(&schema_before)
                            .map(|f| f.name)
                            .unwrap_or_default(),
                    )),
                    _ => None,
                })
                .collect();

            let mut n_having_aggs = 0;
            let updated_having = having_expr.map_expr(|e| {
                if !matches!(&e, Expr::Agg(_) | Expr::Len) {
                    return e;
                }
                let name = agg_to_name
                    .iter()
                    .find_map(|(expr, n)| (*expr == e).then(|| n.clone()))
                    .unwrap_or_else(|| {
                        let n = format_pl_smallstr!("__POLARS_HAVING_{n_having_aggs}");
                        aggregation_projection.push(e.clone().alias(n.clone()));
                        agg_to_name.push((e.clone(), n.clone()));
                        n_having_aggs += 1;
                        n
                    });
                col(name)
            });
            Some(updated_having)
        } else {
            None
        };

        // Apply HAVING filter after aggregation
        let mut aggregated = lf.group_by(group_by_keys).agg(&aggregation_projection);
        if let Some(filter_expr) = having_filter {
            aggregated = aggregated.filter(filter_expr);
        }

        let projection_schema =
            expressions_to_schema(projections, &schema_before, |duplicate_name: &str| {
                format!("group_by aggregations contained duplicate output name '{duplicate_name}'")
            })?;

        // A final projection to get the proper order and any deferred transforms/aliases
        // (will also drop any temporary columns created for the HAVING post-filter).
        let final_projection = projection_schema
            .iter_names()
            .zip(projections.iter().zip(&projection_matches_group_key))
            .map(|(name, (projection_expr, &matches_group_key))| {
                if let Some(expr) = projection_overrides.get(name.as_str()) {
                    expr.clone()
                } else if let Some(aliased_name) = aliased_aggregations.get(name) {
                    col(aliased_name.clone()).alias(name.clone())
                } else if group_by_keys_schema.get(name).is_some() && matches_group_key {
                    col(name.clone())
                } else if group_by_keys_schema.get(name).is_some()
                    || projection_aliases.contains(name.as_str())
                    || group_key_aliases.contains(name.as_str())
                {
                    if has_expr(projection_expr, |e| {
                        matches!(e, Expr::Agg(_) | Expr::Len | Expr::Over { .. })
                    }) {
                        col(name.clone())
                    } else {
                        projection_expr.clone()
                    }
                } else {
                    col(name.clone())
                }
            })
            .collect::<Vec<_>>();

        // Include original GROUP BY columns for ORDER BY access (if aliased).
        let mut output_projection = final_projection;
        for key_name in group_by_keys_schema.iter_names() {
            if !projection_schema.contains(key_name) {
                // Original col name not in output - add for ORDER BY access
                output_projection.push(col(key_name.clone()));
            } else if group_by_keys.iter().any(|k| is_simple_col_ref(k, key_name)) {
                // Original col name in output - check if cross-aliased
                let is_cross_aliased = projections.iter().any(|p| {
                    p.to_field(&schema_before).is_ok_and(|f| f.name == key_name)
                        && !is_simple_col_ref(p, key_name)
                });
                if is_cross_aliased {
                    // Add original name under a prefixed alias for subsequent ORDER BY resolution
                    let internal_name = format_pl_smallstr!("__POLARS_ORIG_{}", key_name);
                    output_projection.push(col(key_name.clone()).alias(internal_name));
                }
            }
        }
        Ok(aggregated.select(&output_projection))
    }

    fn process_limit_offset(
        &self,
        lf: LazyFrame,
        limit_clause: &Option<LimitClause>,
        fetch: &Option<Fetch>,
    ) -> PolarsResult<LazyFrame> {
        // Extract limit and offset from LimitClause
        let (limit, offset) = match limit_clause {
            Some(LimitClause::LimitOffset {
                limit,
                offset,
                limit_by,
            }) => {
                if !limit_by.is_empty() {
                    // TODO: might be able to support as an aggregate `top_k_by` operation?
                    //  (https://clickhouse.com/docs/sql-reference/statements/select/limit-by)
                    polars_bail!(SQLSyntax: "`LIMIT <n> BY <exprs>` clause is not supported");
                }
                (limit.as_ref(), offset.as_ref().map(|o| &o.value))
            },
            Some(LimitClause::OffsetCommaLimit { offset, limit }) => (Some(limit), Some(offset)),
            None => (None, None),
        };

        // Handle FETCH clause (alternative to LIMIT, mutually exclusive)
        let limit = match (fetch, limit) {
            (Some(fetch), None) => fetch.quantity.as_ref(),
            (Some(_), Some(_)) => {
                polars_bail!(SQLSyntax: "cannot use both `LIMIT` and `FETCH` in the same query")
            },
            (None, limit) => limit,
        };

        // Apply limit and/or offset
        match (offset, limit) {
            (
                Some(SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(offset, _),
                    ..
                })),
                Some(SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(limit, _),
                    ..
                })),
            ) => Ok(lf.slice(
                offset
                    .parse()
                    .map_err(|e| polars_err!(SQLInterface: "OFFSET conversion error: {}", e))?,
                limit.parse().map_err(
                    |e| polars_err!(SQLInterface: "LIMIT/FETCH conversion error: {}", e),
                )?,
            )),
            (
                Some(SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(offset, _),
                    ..
                })),
                None,
            ) => Ok(lf.slice(
                offset
                    .parse()
                    .map_err(|e| polars_err!(SQLInterface: "OFFSET conversion error: {}", e))?,
                IdxSize::MAX,
            )),
            (
                None,
                Some(SQLExpr::Value(ValueWithSpan {
                    value: SQLValue::Number(limit, _),
                    ..
                })),
            ) => {
                Ok(lf.limit(limit.parse().map_err(
                    |e| polars_err!(SQLInterface: "LIMIT/FETCH conversion error: {}", e),
                )?))
            },
            (None, None) => Ok(lf),
            _ => polars_bail!(
                SQLSyntax: "non-numeric arguments for LIMIT/OFFSET/FETCH are not supported",
            ),
        }
    }

    fn process_qualified_wildcard(
        &mut self,
        ObjectName(idents): &ObjectName,
        options: &WildcardAdditionalOptions,
        modifiers: &mut SelectModifiers,
        schema: Option<&Schema>,
    ) -> PolarsResult<Vec<Expr>> {
        let mut idents_with_wildcard: Vec<Ident> = idents
            .iter()
            .filter_map(|p| p.as_ident().cloned())
            .collect();
        idents_with_wildcard.push(Ident::new("*"));

        let exprs = resolve_compound_identifier(self, &idents_with_wildcard, schema)?;
        self.process_wildcard_additional_options(exprs, options, modifiers, schema)
    }

    fn process_wildcard_additional_options(
        &mut self,
        exprs: Vec<Expr>,
        options: &WildcardAdditionalOptions,
        modifiers: &mut SelectModifiers,
        schema: Option<&Schema>,
    ) -> PolarsResult<Vec<Expr>> {
        if options.opt_except.is_some() && options.opt_exclude.is_some() {
            polars_bail!(SQLInterface: "EXCLUDE and EXCEPT wildcard options cannot be used together (prefer EXCLUDE)")
        } else if options.opt_exclude.is_some() && options.opt_ilike.is_some() {
            polars_bail!(SQLInterface: "EXCLUDE and ILIKE wildcard options cannot be used together")
        }

        // SELECT * EXCLUDE
        if let Some(items) = &options.opt_exclude {
            match items {
                ExcludeSelectItem::Single(ident) => {
                    modifiers.exclude.insert(ident.value.clone());
                },
                ExcludeSelectItem::Multiple(idents) => {
                    modifiers
                        .exclude
                        .extend(idents.iter().map(|i| i.value.clone()));
                },
            };
        }

        // SELECT * EXCEPT
        if let Some(items) = &options.opt_except {
            modifiers.exclude.insert(items.first_element.value.clone());
            modifiers
                .exclude
                .extend(items.additional_elements.iter().map(|i| i.value.clone()));
        }

        // SELECT * ILIKE
        if let Some(item) = &options.opt_ilike {
            let rx = regex::escape(item.pattern.as_str())
                .replace('%', ".*")
                .replace('_', ".");

            modifiers.ilike = Some(
                polars_utils::regex_cache::compile_regex(format!("^(?is){rx}$").as_str()).unwrap(),
            );
        }

        // SELECT * RENAME
        if let Some(items) = &options.opt_rename {
            let renames = match items {
                RenameSelectItem::Single(rename) => std::slice::from_ref(rename),
                RenameSelectItem::Multiple(renames) => renames.as_slice(),
            };
            for rn in renames {
                let before = PlSmallStr::from_str(rn.ident.value.as_str());
                let after = PlSmallStr::from_str(rn.alias.value.as_str());
                if before != after {
                    modifiers.rename.insert(before, after);
                }
            }
        }

        // SELECT * REPLACE
        if let Some(replacements) = &options.opt_replace {
            for rp in &replacements.items {
                let replacement_expr = parse_sql_expr(&rp.expr, self, schema);
                modifiers
                    .replace
                    .push(replacement_expr?.alias(rp.column_name.value.as_str()));
            }
        }
        Ok(exprs)
    }

    fn rename_columns_from_table_alias(
        &mut self,
        mut lf: LazyFrame,
        alias: &TableAlias,
    ) -> PolarsResult<LazyFrame> {
        if alias.columns.is_empty() {
            Ok(lf)
        } else {
            let schema = self.get_frame_schema(&mut lf)?;
            if alias.columns.len() != schema.len() {
                polars_bail!(
                    SQLSyntax: "number of columns ({}) in alias '{}' does not match the number of columns in the table/query ({})",
                    alias.columns.len(), alias.name.value, schema.len()
                )
            } else {
                let existing_columns: Vec<_> = schema.iter_names().collect();
                let new_columns: Vec<_> =
                    alias.columns.iter().map(|c| c.name.value.clone()).collect();
                Ok(lf.rename(existing_columns, new_columns, true))
            }
        }
    }
}

impl SQLContext {
    /// Create a new SQLContext from a table map. For internal use only
    pub fn new_from_table_map(table_map: PlHashMap<String, LazyFrame>) -> Self {
        Self {
            table_map: Arc::new(RwLock::new(table_map)),
            ..Default::default()
        }
    }
}

fn expand_exprs(expr: Expr, schema: &SchemaRef) -> Vec<Expr> {
    match expr {
        Expr::Column(nm) if is_regex_colname(nm.as_str()) => {
            let re = polars_utils::regex_cache::compile_regex(&nm).unwrap();
            schema
                .iter_names()
                .filter(|name| re.is_match(name))
                .map(|name| col(name.clone()))
                .collect::<Vec<_>>()
        },
        Expr::Selector(s) => s
            .into_columns(schema, &Default::default())
            .unwrap()
            .into_iter()
            .map(col)
            .collect::<Vec<_>>(),
        _ => vec![expr],
    }
}

fn is_regex_colname(nm: &str) -> bool {
    nm.starts_with('^') && nm.ends_with('$')
}

/// Extract column names from a USING clause in a JoinOperator (if present).
fn get_using_cols(op: &JoinOperator) -> Option<impl Iterator<Item = String> + '_> {
    use JoinOperator::*;
    match op {
        Join(JoinConstraint::Using(cols))
        | Inner(JoinConstraint::Using(cols))
        | Left(JoinConstraint::Using(cols))
        | LeftOuter(JoinConstraint::Using(cols))
        | Right(JoinConstraint::Using(cols))
        | RightOuter(JoinConstraint::Using(cols))
        | FullOuter(JoinConstraint::Using(cols))
        | Semi(JoinConstraint::Using(cols))
        | Anti(JoinConstraint::Using(cols))
        | LeftSemi(JoinConstraint::Using(cols))
        | LeftAnti(JoinConstraint::Using(cols))
        | RightSemi(JoinConstraint::Using(cols))
        | RightAnti(JoinConstraint::Using(cols)) => Some(cols.iter().filter_map(|c| {
            c.0.first()
                .and_then(|p| p.as_ident())
                .map(|i| i.value.clone())
        })),
        _ => None,
    }
}

/// Extract the table name (or alias) from a TableFactor.
fn get_table_name(factor: &TableFactor) -> Option<String> {
    match factor {
        TableFactor::Table { name, alias, .. } => {
            alias.as_ref().map(|a| a.name.value.clone()).or_else(|| {
                name.0
                    .last()
                    .and_then(|p| p.as_ident())
                    .map(|i| i.value.clone())
            })
        },
        TableFactor::Derived { alias, .. }
        | TableFactor::NestedJoin { alias, .. }
        | TableFactor::TableFunction { alias, .. } => alias.as_ref().map(|a| a.name.value.clone()),
        _ => None,
    }
}

/// Check if an expression is a simple column reference (with optional alias) to the given name.
fn is_simple_col_ref(expr: &Expr, col_name: &PlSmallStr) -> bool {
    match expr {
        Expr::Column(n) => n == col_name,
        Expr::Alias(inner, _) => matches!(inner.as_ref(), Expr::Column(n) if n == col_name),
        _ => false,
    }
}

/// Strip the outer alias from an expression (if present) for expression equality comparison.
fn strip_outer_alias(expr: &Expr) -> Expr {
    if let Expr::Alias(inner, _) = expr {
        inner.as_ref().clone()
    } else {
        expr.clone()
    }
}

/// Resolve a SELECT alias to its underlying expression (for use in GROUP BY).
///
/// Returns the expression WITH alias if the name matches a projection alias and is NOT a column
/// that exists in the schema; otherwise returns `None` to use the default/standard resolution.
fn resolve_select_alias(name: &str, projections: &[Expr], schema: &Schema) -> Option<Expr> {
    // Original columns take precedence over SELECT aliases
    if schema.contains(name) {
        return None;
    }
    // Find a projection with this alias and return its expression (preserving the alias)
    projections.iter().find_map(|p| match p {
        Expr::Alias(inner, alias) if alias.as_str() == name => {
            Some(inner.as_ref().clone().alias(alias.clone()))
        },
        _ => None,
    })
}

/// Check if all columns referred to in a Polars expression exist in the given Schema.
fn expr_cols_all_in_schema(expr: &Expr, schema: &Schema) -> bool {
    let mut found_cols = false;
    let mut all_in_schema = true;
    for e in expr.into_iter() {
        if let Expr::Column(name) = e {
            found_cols = true;
            if !schema.contains(name.as_str()) {
                all_in_schema = false;
                break;
            }
        }
    }
    found_cols && all_in_schema
}

/// Determine which parsed join expressions actually belong in `left_om` and which in `right_on`.
///
/// This needs to be handled carefully because in SQL joins you can write "join on" constraints
/// either way round, and in joins with more than two tables you can also join against an earlier
/// table (e.g.: you could be joining `df1` to `df2` to `df3`, but the final join condition where
/// we join `df2` to `df3` could refer to `df1.a = df3.b`; this takes a little more work to
/// resolve as our native `join` function operates on only two tables at a time.
fn determine_left_right_join_on(
    ctx: &mut SQLContext,
    expr_left: &SQLExpr,
    expr_right: &SQLExpr,
    tbl_left: &TableInfo,
    tbl_right: &TableInfo,
    join_schema: &Schema,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    // parse, removing any aliases that may have been added by `resolve_column`
    // (called inside `parse_sql_expr`) as we need the actual/underlying col
    let left_on = match parse_sql_expr(expr_left, ctx, Some(join_schema))? {
        Expr::Alias(inner, _) => Arc::unwrap_or_clone(inner),
        e => e,
    };
    let right_on = match parse_sql_expr(expr_right, ctx, Some(join_schema))? {
        Expr::Alias(inner, _) => Arc::unwrap_or_clone(inner),
        e => e,
    };

    // ------------------------------------------------------------------
    // simple/typical case: can fully resolve SQL-level table references
    // ------------------------------------------------------------------
    let left_refs = (
        expr_refers_to_table(expr_left, &tbl_left.name),
        expr_refers_to_table(expr_left, &tbl_right.name),
    );
    let right_refs = (
        expr_refers_to_table(expr_right, &tbl_left.name),
        expr_refers_to_table(expr_right, &tbl_right.name),
    );
    // if the SQL-level references unambiguously indicate table ownership, we're done
    match (left_refs, right_refs) {
        // standard: left expr → left table, right expr → right table
        ((true, false), (false, true)) => return Ok((vec![left_on], vec![right_on])),
        // reversed: left expr → right table, right expr → left table
        ((false, true), (true, false)) => return Ok((vec![right_on], vec![left_on])),
        // unsupported: one side references *both* tables
        ((true, true), _) | (_, (true, true)) if tbl_left.name != tbl_right.name => {
            polars_bail!(
               SQLInterface: "unsupported join condition: {} side references both '{}' and '{}'",
               if left_refs.0 && left_refs.1 {
                    "left"
                } else {
                    "right"
                }, tbl_left.name, tbl_right.name
            )
        },
        // fall through to the more involved col/ref resolution
        _ => {},
    }

    // ------------------------------------------------------------------
    // more involved: additionally employ schema-based column resolution
    // (applies to unqualified columns and/or chained joins)
    // ------------------------------------------------------------------
    let left_on_cols_in = (
        expr_cols_all_in_schema(&left_on, &tbl_left.schema),
        expr_cols_all_in_schema(&left_on, &tbl_right.schema),
    );
    let right_on_cols_in = (
        expr_cols_all_in_schema(&right_on, &tbl_left.schema),
        expr_cols_all_in_schema(&right_on, &tbl_right.schema),
    );
    match (left_on_cols_in, right_on_cols_in) {
        // each expression's columns exist in exactly one schema
        ((true, false), (false, true)) => Ok((vec![left_on], vec![right_on])),
        ((false, true), (true, false)) => Ok((vec![right_on], vec![left_on])),
        // one expression in both, other only in one; prefer the unique one
        ((true, true), (true, false)) => Ok((vec![right_on], vec![left_on])),
        ((true, true), (false, true)) => Ok((vec![left_on], vec![right_on])),
        ((true, false), (true, true)) => Ok((vec![left_on], vec![right_on])),
        ((false, true), (true, true)) => Ok((vec![right_on], vec![left_on])),
        // pass through as-is
        _ => Ok((vec![left_on], vec![right_on])),
    }
}

fn process_join_on(
    ctx: &mut SQLContext,
    sql_expr: &SQLExpr,
    tbl_left: &TableInfo,
    tbl_right: &TableInfo,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    match sql_expr {
        SQLExpr::BinaryOp { left, op, right } => match op {
            BinaryOperator::And => {
                let (mut left_i, mut right_i) = process_join_on(ctx, left, tbl_left, tbl_right)?;
                let (mut left_j, mut right_j) = process_join_on(ctx, right, tbl_left, tbl_right)?;
                left_i.append(&mut left_j);
                right_i.append(&mut right_j);
                Ok((left_i, right_i))
            },
            BinaryOperator::Eq => {
                // establish unified schema with cols from both tables; needed for multi/chained
                // joins where suffixed intermediary/joined cols aren't in an existing schema.
                let mut join_schema =
                    Schema::with_capacity(tbl_left.schema.len() + tbl_right.schema.len());
                for (name, dtype) in tbl_left.schema.iter() {
                    join_schema.insert_at_index(join_schema.len(), name.clone(), dtype.clone())?;
                }
                for (name, dtype) in tbl_right.schema.iter() {
                    if !join_schema.contains(name) {
                        join_schema.insert_at_index(
                            join_schema.len(),
                            name.clone(),
                            dtype.clone(),
                        )?;
                    }
                }
                determine_left_right_join_on(ctx, left, right, tbl_left, tbl_right, &join_schema)
            },
            _ => polars_bail!(
                // TODO: should be able to support more operators later (via `join_where`?)
                SQLInterface: "only equi-join constraints (combined with 'AND') are currently supported; found op = '{:?}'", op
            ),
        },
        SQLExpr::Nested(expr) => process_join_on(ctx, expr, tbl_left, tbl_right),
        _ => polars_bail!(
            SQLInterface: "only equi-join constraints are currently supported; found expression = {:?}", sql_expr
        ),
    }
}

fn process_join_constraint(
    constraint: &JoinConstraint,
    tbl_left: &TableInfo,
    tbl_right: &TableInfo,
    ctx: &mut SQLContext,
) -> PolarsResult<(Vec<Expr>, Vec<Expr>)> {
    match constraint {
        JoinConstraint::On(expr @ SQLExpr::BinaryOp { .. }) => {
            process_join_on(ctx, expr, tbl_left, tbl_right)
        },
        JoinConstraint::Using(idents) if !idents.is_empty() => {
            let using: Vec<Expr> = idents
                .iter()
                .map(|ObjectName(parts)| {
                    if parts.len() != 1 {
                        polars_bail!(SQLSyntax: "JOIN \"USING\" clause expects simple column names, not qualified names");
                    }
                    match parts[0].as_ident() {
                        Some(ident) => Ok(col(ident.value.as_str())),
                        None => polars_bail!(SQLSyntax: "JOIN \"USING\" clause expects identifiers, not functions"),
                    }
                })
                .collect::<PolarsResult<Vec<_>>>()?;
            Ok((using.clone(), using))
        },
        JoinConstraint::Natural => {
            let left_names = tbl_left.schema.iter_names().collect::<PlHashSet<_>>();
            let right_names = tbl_right.schema.iter_names().collect::<PlHashSet<_>>();
            let on: Vec<Expr> = left_names
                .intersection(&right_names)
                .map(|&name| col(name.clone()))
                .collect();
            if on.is_empty() {
                polars_bail!(SQLInterface: "no common columns found for NATURAL JOIN")
            }
            Ok((on.clone(), on))
        },
        _ => polars_bail!(SQLInterface: "unsupported SQL join constraint:\n{:?}", constraint),
    }
}

/// Extract table identifiers referenced in a SQL query; uses a visitor to
/// collect all table names that appear in FROM clauses, JOINs, TABLE refs
/// in set operations, and subqueries.
pub fn extract_table_identifiers(
    query: &str,
    include_schema: bool,
    unique: bool,
) -> PolarsResult<Vec<String>> {
    let mut parser = Parser::new(&GenericDialect);
    parser = parser.with_options(ParserOptions {
        trailing_commas: true,
        ..Default::default()
    });
    let ast = parser
        .try_with_sql(query)
        .map_err(to_sql_interface_err)?
        .parse_statements()
        .map_err(to_sql_interface_err)?;

    let mut collector = TableIdentifierCollector {
        include_schema,
        ..Default::default()
    };
    for stmt in &ast {
        let _ = stmt.visit(&mut collector);
    }
    Ok(if unique {
        collector
            .tables
            .into_iter()
            .collect::<PlIndexSet<_>>()
            .into_iter()
            .collect()
    } else {
        collector.tables
    })
}

bitflags::bitflags! {
    /// Bitfield indicating whether there exists a projection with the specified height behavior.
    ///
    /// Used to help determine whether to execute projections in `select()` or `with_columns()`
    /// context.
    #[derive(PartialEq)]
    struct ExprSqlProjectionHeightBehavior: u8 {
        /// Maintains the height of input column(s)
        const MaintainsColumn = 1 << 0;
        /// Height is independent of input, e.g.:
        /// * expressions that change length: e.g. slice, explode, filter, gather etc.
        /// * aggregations: count(*), first(), sum() etc.
        const Independent = 1 << 1;
        /// "Inherits" the height of the context, e.g.:
        /// * Scalar literals
        const InheritsContext = 1 << 2;
    }
}

impl ExprSqlProjectionHeightBehavior {
    fn identify_from_expr(expr: &Expr) -> Self {
        let mut has_column = false;
        let mut has_independent = false;

        for e in expr.into_iter() {
            use Expr::*;
            has_column |= matches!(e, Column(_) | Selector(_));
            has_independent |= match e {
                // @TODO: This is broken now with functions.
                AnonymousFunction { options, .. } => {
                    options.returns_scalar() || !options.is_length_preserving()
                },
                Literal(v) => !v.is_scalar(),
                Explode { .. } | Filter { .. } | Gather { .. } | Slice { .. } => true,
                Agg { .. } | Len => true,
                _ => false,
            }
        }
        if has_independent {
            Self::Independent
        } else if has_column {
            Self::MaintainsColumn
        } else {
            Self::InheritsContext
        }
    }
}
