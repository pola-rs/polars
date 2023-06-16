use std::cell::RefCell;
use std::collections::BTreeSet;

use polars_arrow::error::to_compute_err;
use polars_core::prelude::*;
use polars_lazy::prelude::*;
use polars_plan::prelude::*;
use polars_plan::utils::expressions_to_schema;
use sqlparser::ast::{
    Distinct, ExcludeSelectItem, Expr as SqlExpr, FunctionArg, JoinOperator, ObjectName,
    ObjectType, Offset, OrderByExpr, Query, Select, SelectItem, SetExpr, SetOperator,
    SetQuantifier, Statement, TableAlias, TableFactor, TableWithJoins, Value as SQLValue,
    WildcardAdditionalOptions,
};
use sqlparser::dialect::GenericDialect;
use sqlparser::parser::{Parser, ParserOptions};

use crate::sql_expr::{parse_sql_expr, process_join_constraint};
use crate::table_functions::PolarsTableFunctions;

/// The SQLContext is the main entry point for executing SQL queries.
#[derive(Default, Clone)]
pub struct SQLContext {
    pub(crate) table_map: PlHashMap<String, LazyFrame>,
    cte_map: RefCell<PlHashMap<String, LazyFrame>>,
}

impl SQLContext {
    /// Create a new SQLContext
    /// ```rust
    /// # use polars_sql::SQLContext;
    /// # fn main() {
    /// let ctx = SQLContext::new();
    /// # }
    /// ```
    pub fn new() -> Self {
        Self {
            table_map: PlHashMap::new(),
            cte_map: RefCell::new(PlHashMap::new()),
        }
    }

    /// Get the names of all registered tables, in sorted order.
    pub fn get_tables(&self) -> Vec<String> {
        let mut tables = Vec::from_iter(self.table_map.keys().cloned());
        tables.sort_unstable();
        tables
    }

    /// Register a LazyFrame as a table in the SQLContext.
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
    pub fn register(&mut self, name: &str, lf: LazyFrame) {
        self.table_map.insert(name.to_owned(), lf);
    }

    /// Unregister a LazyFrame table from the SQLContext.
    pub fn unregister(&mut self, name: &str) {
        self.table_map.remove(&name.to_owned());
    }

    /// Execute a SQL query, returning a LazyFrame.
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
    /// assert!(sql_df.frame_equal(&df));
    /// # }
    ///```
    pub fn execute(&mut self, query: &str) -> PolarsResult<LazyFrame> {
        let mut parser = Parser::new(&GenericDialect);
        parser = parser.with_options(ParserOptions {
            trailing_commas: true,
        });

        let ast = parser
            .try_with_sql(query)
            .map_err(to_compute_err)?
            .parse_statements()
            .map_err(to_compute_err)?;
        polars_ensure!(ast.len() == 1, ComputeError: "One and only one statement at a time please");
        let res = self.execute_statement(ast.get(0).unwrap());
        // every execution should clear the cte map
        self.cte_map.borrow_mut().clear();
        res
    }
}

impl SQLContext {
    fn register_cte(&mut self, name: &str, lf: LazyFrame) {
        self.cte_map.borrow_mut().insert(name.to_owned(), lf);
    }

    fn get_table_from_current_scope(&mut self, name: &str) -> Option<LazyFrame> {
        if let Some(lf) = self.table_map.get(name) {
            Some(lf.clone())
        } else {
            self.cte_map.borrow().get(name).cloned()
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
            _ => polars_bail!(
                ComputeError: "SQL statement type {:?} is not supported", ast,
            ),
        })
    }

    pub(crate) fn execute_query(&mut self, query: &Query) -> PolarsResult<LazyFrame> {
        self.register_ctes(query)?;

        let lf = self.process_set_expr(&query.body, query)?;

        self.process_limit_offset(lf, &query.limit, &query.offset)
    }

    fn process_set_expr(&mut self, expr: &SetExpr, query: &Query) -> PolarsResult<LazyFrame> {
        match expr {
            SetExpr::Select(select_stmt) => self.execute_select(select_stmt, query),
            SetExpr::Query(query) => self.execute_query(query),
            SetExpr::SetOperation {
                op: SetOperator::Union,
                set_quantifier,
                left,
                right,
            } => self.process_union(left, right, set_quantifier, query),
            SetExpr::SetOperation { op, .. } => {
                polars_bail!(InvalidOperation: "{} operation not yet supported", op)
            }
            op => polars_bail!(InvalidOperation: "{} operation not yet supported", op),
        }
    }

    fn process_union(
        &mut self,
        left: &SetExpr,
        right: &SetExpr,
        quantifier: &SetQuantifier,
        query: &Query,
    ) -> PolarsResult<LazyFrame> {
        let left = self.process_set_expr(left, query)?;
        let right = self.process_set_expr(right, query)?;
        let concatenated = polars_lazy::dsl::concat(
            vec![left, right],
            UnionArgs {
                parallel: true,
                ..Default::default()
            },
        );
        match quantifier {
            // UNION ALL
            SetQuantifier::All => concatenated,
            // UNION DISTINCT | UNION
            _ => concatenated.map(|lf| lf.unique(None, UniqueKeepStrategy::Any)),
        }
    }
    // EXPLAIN SELECT * FROM DF
    fn execute_explain(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        match stmt {
            Statement::Explain { statement, .. } => {
                let lf = self.execute_statement(statement)?;
                let plan = lf.describe_optimized_plan()?;
                let mut plan = plan.split('\n').collect::<Series>();
                plan.rename("Logical Plan");

                let df = DataFrame::new(vec![plan])?;
                Ok(df.lazy())
            }
            _ => unreachable!(),
        }
    }

    /// SHOW TABLES
    fn execute_show_tables(&mut self, _: &Statement) -> PolarsResult<LazyFrame> {
        let tables = Series::new("name", self.get_tables());
        let df = DataFrame::new(vec![tables])?;
        Ok(df.lazy())
    }

    fn execute_drop_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        match stmt {
            Statement::Drop { names, .. } => {
                for name in names {
                    self.table_map.remove(&name.to_string());
                }
                Ok(DataFrame::empty().lazy())
            }
            _ => unreachable!(),
        }
    }

    fn register_ctes(&mut self, query: &Query) -> PolarsResult<()> {
        if let Some(with) = &query.with {
            if with.recursive {
                polars_bail!(ComputeError: "Recursive CTEs are not supported")
            }
            for cte in &with.cte_tables {
                let cte_name = cte.alias.name.to_string();
                let cte_lf = self.execute_query(&cte.query)?;
                self.register_cte(&cte_name, cte_lf);
            }
        }
        Ok(())
    }

    /// execute the 'FROM' part of the query
    fn execute_from_statement(&mut self, tbl_expr: &TableWithJoins) -> PolarsResult<LazyFrame> {
        let (tbl_name, mut lf) = self.get_table(&tbl_expr.relation)?;
        if !tbl_expr.joins.is_empty() {
            for tbl in &tbl_expr.joins {
                let (join_tbl_name, join_tbl) = self.get_table(&tbl.relation)?;
                match &tbl.join_operator {
                    JoinOperator::Inner(constraint) => {
                        let (left_on, right_on) =
                            process_join_constraint(constraint, &tbl_name, &join_tbl_name)?;
                        lf = lf.inner_join(join_tbl, left_on, right_on)
                    }
                    JoinOperator::LeftOuter(constraint) => {
                        let (left_on, right_on) =
                            process_join_constraint(constraint, &tbl_name, &join_tbl_name)?;
                        lf = lf.left_join(join_tbl, left_on, right_on)
                    }
                    JoinOperator::FullOuter(constraint) => {
                        let (left_on, right_on) =
                            process_join_constraint(constraint, &tbl_name, &join_tbl_name)?;
                        lf = lf.outer_join(join_tbl, left_on, right_on)
                    }
                    JoinOperator::CrossJoin => lf = lf.cross_join(join_tbl),
                    join_type => {
                        polars_bail!(
                            InvalidOperation:
                            "join type '{:?}' not yet supported by polars-sql", join_type
                        );
                    }
                }
            }
        };

        Ok(lf)
    }
    /// execute the 'SELECT' part of the query
    fn execute_select(&mut self, select_stmt: &Select, query: &Query) -> PolarsResult<LazyFrame> {
        // Determine involved dataframe
        // Implicit join require some more work in query parsers, Explicit join are preferred for now.
        let sql_tbl: &TableWithJoins = select_stmt
            .from
            .get(0)
            .ok_or_else(|| polars_err!(ComputeError: "no table name provided in query"))?;

        let mut lf = self.execute_from_statement(sql_tbl)?;
        let mut contains_wildcard = false;

        // Filter Expression
        lf = match select_stmt.selection.as_ref() {
            Some(expr) => {
                let filter_expression = parse_sql_expr(expr, self)?;
                lf.filter(filter_expression)
            }
            None => lf,
        };

        // Column Projections
        let projections: Vec<_> = select_stmt
            .projection
            .iter()
            .map(|select_item| {
                Ok(match select_item {
                    SelectItem::UnnamedExpr(expr) => parse_sql_expr(expr, self)?,
                    SelectItem::ExprWithAlias { expr, alias } => {
                        let expr = parse_sql_expr(expr, self)?;
                        expr.alias(&alias.value)
                    }
                    SelectItem::QualifiedWildcard(oname, wildcard_options) => {
                        self.process_qualified_wildcard(oname, wildcard_options)?
                    }
                    SelectItem::Wildcard(wildcard_options) => {
                        contains_wildcard = true;
                        let e = col("*");
                        self.process_wildcard_additional_options(e, wildcard_options)?
                    }
                })
            })
            .collect::<PolarsResult<_>>()?;

        // Check for group by
        // After projection since there might be number.
        let groupby_keys: Vec<Expr> = select_stmt
            .group_by
            .iter()
            .map(|e| match e {
                SqlExpr::Value(SQLValue::Number(idx, _)) => {
                    let idx = match idx.parse::<usize>() {
                        Ok(0) | Err(_) => Err(polars_err!(
                            ComputeError:
                            "groupby error: a positive number or an expression expected, got {}",
                            idx
                        )),
                        Ok(idx) => Ok(idx),
                    }?;
                    Ok(projections[idx].clone())
                }
                SqlExpr::Value(_) => Err(polars_err!(
                    ComputeError:
                    "groupby error: a positive number or an expression expected",
                )),
                _ => parse_sql_expr(e, self),
            })
            .collect::<PolarsResult<_>>()?;

        if groupby_keys.is_empty() {
            lf = lf.select(projections)
        } else {
            lf = self.process_groupby(lf, contains_wildcard, &groupby_keys, &projections)?;

            // Apply optional 'having' clause, post-aggregation
            lf = match select_stmt.having.as_ref() {
                Some(expr) => lf.filter(parse_sql_expr(expr, self)?),
                None => lf,
            };
        };

        // Apply optional 'distinct' clause
        lf = match &select_stmt.distinct {
            Some(Distinct::Distinct) => lf.unique(None, UniqueKeepStrategy::Any),
            Some(Distinct::On(exprs)) => {
                // TODO: support exprs in `unique` see https://github.com/pola-rs/polars/issues/5760
                let cols = exprs
                    .iter()
                    .map(|e| {
                        let expr = parse_sql_expr(e, self)?;
                        if let Expr::Column(name) = expr {
                            Ok(name.to_string())
                        } else {
                            Err(polars_err!(
                                ComputeError:
                                "DISTINCT ON only supports column names"
                            ))
                        }
                    })
                    .collect::<PolarsResult<Vec<_>>>()?;

                // DISTINCT ON applies the ORDER BY before the operation.
                if !query.order_by.is_empty() {
                    lf = self.process_order_by(lf, &query.order_by)?;
                }
                return Ok(lf.unique_stable(Some(cols), UniqueKeepStrategy::First));
            }
            None => lf,
        };

        if query.order_by.is_empty() {
            Ok(lf)
        } else {
            self.process_order_by(lf, &query.order_by)
        }
    }

    fn execute_create_table(&mut self, stmt: &Statement) -> PolarsResult<LazyFrame> {
        if let Statement::CreateTable {
            if_not_exists,
            name,
            query,
            ..
        } = stmt
        {
            let tbl_name = name.0.get(0).unwrap().value.as_str();
            // CREATE TABLE IF NOT EXISTS
            if *if_not_exists && self.table_map.contains_key(tbl_name) {
                polars_bail!(ComputeError: "relation {} already exists", tbl_name);
                // CREATE OR REPLACE TABLE
            }
            if let Some(query) = query {
                let lf = self.execute_query(query)?;
                self.register(tbl_name, lf);
                let out = df! {
                    "Response" => ["Create Table"]
                }
                .unwrap()
                .lazy();
                Ok(out)
            } else {
                polars_bail!(ComputeError: "only CREATE TABLE AS SELECT is supported");
            }
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
                    return self.execute_tbl_function(name, alias, args);
                }
                let tbl_name = name.0.get(0).unwrap().value.as_str();
                if let Some(lf) = self.get_table_from_current_scope(tbl_name) {
                    Ok((tbl_name.to_string(), lf))
                } else {
                    polars_bail!(ComputeError: "relation '{}' was not found", tbl_name);
                }
            }
            // Support bare table, optional with alias for now
            _ => polars_bail!(ComputeError: "not implemented"),
        }
    }

    fn execute_tbl_function(
        &mut self,
        name: &ObjectName,
        alias: &Option<TableAlias>,
        args: &[FunctionArg],
    ) -> PolarsResult<(String, LazyFrame)> {
        let tbl_fn = name.0.get(0).unwrap().value.as_str();
        let read_fn = tbl_fn.parse::<PolarsTableFunctions>()?;
        let (tbl_name, lf) = read_fn.execute(args)?;
        let tbl_name = alias
            .as_ref()
            .map(|a| a.name.value.clone())
            .unwrap_or_else(|| tbl_name);

        self.table_map.insert(tbl_name.clone(), lf.clone());

        Ok((tbl_name, lf))
    }

    fn process_order_by(&mut self, lf: LazyFrame, ob: &[OrderByExpr]) -> PolarsResult<LazyFrame> {
        let mut by = Vec::with_capacity(ob.len());
        let mut descending = Vec::with_capacity(ob.len());

        for ob in ob {
            by.push(parse_sql_expr(&ob.expr, self)?);
            if let Some(false) = ob.asc {
                descending.push(true)
            } else {
                descending.push(false)
            }
            polars_ensure!(
                ob.nulls_first.is_none(),
                ComputeError: "nulls first/last is not yet supported",
            );
        }

        Ok(lf.sort_by_exprs(&by, descending, false))
    }

    fn process_groupby(
        &mut self,
        lf: LazyFrame,
        contains_wildcard: bool,
        groupby_keys: &[Expr],
        projections: &[Expr],
    ) -> PolarsResult<LazyFrame> {
        // check groupby and projection due to difference between SQL and polars
        // Return error on wild card, shouldn't process this
        polars_ensure!(
            !contains_wildcard,
            ComputeError: "groupby error: can't process wildcard in groupby"
        );
        let schema_before = lf.schema()?;

        let groupby_keys_schema =
            expressions_to_schema(groupby_keys, &schema_before, Context::Default)?;

        // remove the groupby keys as polars adds those implicitly
        let mut aggregation_projection = Vec::with_capacity(projections.len());
        let mut aliases: BTreeSet<&str> = BTreeSet::new();

        for mut e in projections {
            // if it is a simple expression & has alias,
            // we must defer the aliasing until after the groupby
            if e.clone().meta().is_simple_projection() {
                if let Expr::Alias(expr, name) = e {
                    aliases.insert(name);
                    e = expr
                }
            }

            let field = e.to_field(&schema_before, Context::Default)?;
            if groupby_keys_schema.get(&field.name).is_none() {
                aggregation_projection.push(e.clone())
            }
        }

        let aggregated = lf.groupby(groupby_keys).agg(&aggregation_projection);
        let projection_schema =
            expressions_to_schema(projections, &schema_before, Context::Default)?;
        // a final projection to get the proper order
        let final_projection = projection_schema
            .iter_names()
            .zip(projections)
            .map(|(name, projection_expr)| {
                if groupby_keys_schema.get(name).is_some() || aliases.contains(name.as_str()) {
                    projection_expr.clone()
                } else {
                    col(name)
                }
            })
            .collect::<Vec<_>>();

        Ok(aggregated.select(&final_projection))
    }

    fn process_limit_offset(
        &mut self,
        lf: LazyFrame,
        limit: &Option<SqlExpr>,
        offset: &Option<Offset>,
    ) -> PolarsResult<LazyFrame> {
        match (offset, limit) {
            (
                Some(Offset {
                    value: SqlExpr::Value(SQLValue::Number(offset, _)),
                    ..
                }),
                Some(SqlExpr::Value(SQLValue::Number(limit, _))),
            ) => Ok(lf.slice(
                offset
                    .parse()
                    .map_err(|e| polars_err!(ComputeError: "OFFSET conversion error: {}", e))?,
                limit
                    .parse()
                    .map_err(|e| polars_err!(ComputeError: "LIMIT conversion error: {}", e))?,
            )),
            (
                Some(Offset {
                    value: SqlExpr::Value(SQLValue::Number(offset, _)),
                    ..
                }),
                None,
            ) => Ok(lf.slice(
                offset
                    .parse()
                    .map_err(|e| polars_err!(ComputeError: "OFFSET conversion error: {}", e))?,
                IdxSize::MAX,
            )),
            (None, Some(SqlExpr::Value(SQLValue::Number(limit, _)))) => Ok(lf.limit(
                limit
                    .parse()
                    .map_err(|e| polars_err!(ComputeError: "LIMIT conversion error: {}", e))?,
            )),
            (None, None) => Ok(lf),
            _ => polars_bail!(
                ComputeError: "non-numeric arguments for LIMIT/OFFSET are not supported",
            ),
        }
    }

    fn process_qualified_wildcard(
        &mut self,
        ObjectName(idents): &ObjectName,
        options: &WildcardAdditionalOptions,
    ) -> PolarsResult<Expr> {
        let idents = idents.as_slice();
        let e = match idents {
            [tbl_name] => {
                let lf = self.table_map.get(&tbl_name.value).ok_or_else(|| {
                    polars_err!(
                        ComputeError: "no table named '{}' found",
                        tbl_name
                    )
                })?;
                let schema = lf.schema()?;
                cols(schema.iter_names())
            }
            e => polars_bail!(
                ComputeError: "Invalid wildcard expression: {:?}",
                e
            ),
        };
        self.process_wildcard_additional_options(e, options)
    }

    fn process_wildcard_additional_options(
        &mut self,
        expr: Expr,
        options: &WildcardAdditionalOptions,
    ) -> PolarsResult<Expr> {
        if options.opt_except.is_some() {
            polars_bail!(InvalidOperation: "EXCEPT not supported. Use EXCLUDE instead")
        }
        Ok(match &options.opt_exclude {
            Some(ExcludeSelectItem::Single(ident)) => expr.exclude(vec![&ident.value]),
            Some(ExcludeSelectItem::Multiple(idents)) => {
                expr.exclude(idents.iter().map(|i| &i.value))
            }
            _ => expr,
        })
    }
}

impl SQLContext {
    /// Get internal table map. For internal use only.
    pub fn get_table_map(&self) -> PlHashMap<String, LazyFrame> {
        self.table_map.clone()
    }

    /// Create a new SQLContext from a table map. For internal use only
    pub fn new_from_table_map(table_map: PlHashMap<String, LazyFrame>) -> Self {
        Self {
            table_map,
            cte_map: RefCell::new(PlHashMap::new()),
        }
    }
}
