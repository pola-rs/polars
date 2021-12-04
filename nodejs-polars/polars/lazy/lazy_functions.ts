import {Expr} from "./expr";
import {Series} from "../series";
import {ColumnSelection, ColumnsOrExpr, ExpressionSelection, isSeries, Option} from "../utils";
import pli from "../internals/polars_internal";

/**
 * __A column in a DataFrame.__
 * Can be used to select:
 *   * a single column by name
 *   * all columns by using a wildcard `"*"`
 *   * column by regular expression if the regex starts with `^` and ends with `$`
 * @param col
 * @example
 * ```
 * >>> df = pl.DataFrame({
 * >>> "ham": [1, 2, 3],
 * >>> "hamburger": [11, 22, 33],
 * >>> "foo": [3, 2, 1]})
 * >>> df.select(col("foo"))
 * shape: (3, 1)
 * ╭─────╮
 * │ foo │
 * │ --- │
 * │ i64 │
 * ╞═════╡
 * │ 3   │
 * ├╌╌╌╌╌┤
 * │ 2   │
 * ├╌╌╌╌╌┤
 * │ 1   │
 * ╰─────╯
 * >>> df.select(col("*"))
 * shape: (3, 3)
 * ╭─────┬───────────┬─────╮
 * │ ham ┆ hamburger ┆ foo │
 * │ --- ┆ ---       ┆ --- │
 * │ i64 ┆ i64       ┆ i64 │
 * ╞═════╪═══════════╪═════╡
 * │ 1   ┆ 11        ┆ 3   │
 * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 2   ┆ 22        ┆ 2   │
 * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 3   ┆ 33        ┆ 1   │
 * ╰─────┴───────────┴─────╯
 * >>> df.select(col("^ham.*$"))
 * shape: (3, 2)
 * ╭─────┬───────────╮
 * │ ham ┆ hamburger │
 * │ --- ┆ ---       │
 * │ i64 ┆ i64       │
 * ╞═════╪═══════════╡
 * │ 1   ┆ 11        │
 * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
 * │ 2   ┆ 22        │
 * ├╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌╌╌┤
 * │ 3   ┆ 33        │
 * ╰─────┴───────────╯
 * >>> df.select(col("*").exclude("ham"))
 * shape: (3, 2)
 * ╭───────────┬─────╮
 * │ hamburger ┆ foo │
 * │ ---       ┆ --- │
 * │ i64       ┆ i64 │
 * ╞═══════════╪═════╡
 * │ 11        ┆ 3   │
 * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 22        ┆ 2   │
 * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 33        ┆ 1   │
 * ╰───────────┴─────╯
 * >>> df.select(col(["hamburger", "foo"])
 * shape: (3, 2)
 * ╭───────────┬─────╮
 * │ hamburger ┆ foo │
 * │ ---       ┆ --- │
 * │ i64       ┆ i64 │
 * ╞═══════════╪═════╡
 * │ 11        ┆ 3   │
 * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 22        ┆ 2   │
 * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 33        ┆ 1   │
 * ╰───────────┴─────╯
 * >>> df.select(col(pl.Series(["hamburger", "foo"]))
 * shape: (3, 2)
 * ╭───────────┬─────╮
 * │ hamburger ┆ foo │
 * │ ---       ┆ --- │
 * │ i64       ┆ i64 │
 * ╞═══════════╪═════╡
 * │ 11        ┆ 3   │
 * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 22        ┆ 2   │
 * ├╌╌╌╌╌╌╌╌╌╌╌┼╌╌╌╌╌┤
 * │ 33        ┆ 1   │
 * ╰───────────┴─────╯
 */
export function col(col: string): Expr
export function col(col: string[]): Expr
export function col(col: Series<string>): Expr
export function col(col: string | string[] | Series<string>): Expr {
  if(isSeries(col)) {
    col = col.toArray();
  }
  if(Array.isArray(col)) {
    return Expr(pli.cols({name: col}));
  } else {
    return Expr(pli.col({name: col}));
  }
}

/**
 * Count the number of values in this column.
 */
export function count(column: string): Expr
export function count(column: Series<any>): number
export function count(column: string | Series<any>): Expr | number {
  if(isSeries(column)) {
    return column.len();
  } else {
    return col(column).count();
  }
}
/**
 * Aggregate to list.
 *
 *  Re-exported as `pl.list()`
 * @param name coulmn name
 */
export function toList(name:string): Expr {
  return col(name).list();
}


/**
 * Get the maximum value. Can be used horizontally or vertically.
 * @param column
 */
function max(column: ExpressionSelection): Expr;
function max<T>(column: Series<T>): number | bigint;
function max(column): Expr | number | bigint {

  return null as any;
}
// function max_() {}
// function min() {}
// function min() {}
// function min() {}
// function min_() {}
// function sum() {}
// function sum() {}
// function sum() {}
// function mean() {}
// function mean() {}
// function mean() {}
// function avg() {}
// function avg() {}
// function avg() {}
// function median() {}
// function median() {}
// function median() {}
// function n_unique() {}
// function n_unique() {}
// function n_unique() {}
// function first() {}
// function first() {}
// function first() {}
// function last() {}
// function last() {}
// function last() {}
// function head() {}
// function head() {}
// function head() {}
// function tail() {}
// function tail() {}
// function tail() {}
function lit() {}
function spearman_rank_corr() {}
function pearson_corr() {}
function cov() {}
function map() {}
function apply() {}
function map_binary() {}
function fold() {}
function any() {}
function exclude() {}
function all() {}
function groups() {}
function quantile() {}
function arange() {}
function argsort_by() {}
function _datetime() {}
function _date() {}
function concat_str() {}
function format() {}
function concat_list() {}
function collect_all() {}
function select() {}