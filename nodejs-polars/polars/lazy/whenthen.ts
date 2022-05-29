import {Expr} from "./expr";
import pli from "../internals/polars_internal";


export interface When {
  /** Values to return in case of the predicate being `true`.*/
  then(expr: Expr): WhenThen
}

export interface WhenThen {
  /** Start another when, then, otherwise layer. */
  when(predicate: Expr): WhenThenThen
  /** Values to return in case of the predicate being `false`. */
  otherwise(expr: Expr): Expr
}

export interface WhenThenThen {
  /** Start another when, then, otherwise layer. */
  when(predicate: Expr): WhenThenThen
  /** Values to return in case of the predicate being `true`. */
  then(expr: Expr): WhenThenThen
  /** Values to return in case of the predicate being `false`. */
  otherwise(expr: Expr): Expr
}

function WhenThenThen(_whenthenthen: any): WhenThenThen {
  return {
    when: ({_expr}: Expr): WhenThenThen => WhenThenThen(_whenthenthen.when(_expr)),
    then: ({_expr}: Expr): WhenThenThen => WhenThenThen(_whenthenthen.then(_expr)),
    otherwise: ({_expr}: Expr): Expr => (Expr as any)(_whenthenthen.otherwise(_expr))
  };
}

function WhenThen(_whenthen: any): WhenThen {
  return {
    when: ({_expr}: Expr): WhenThenThen => WhenThenThen(_whenthen.when(_expr)),
    otherwise: ({_expr}: Expr): Expr => (Expr as any)(_whenthen.otherwise(_expr))
  };
}


/**
 * Utility function.
 * @see {@link when}
 */
function When(_when: any): When {
  return {
    then: ({_expr}: Expr): WhenThen => WhenThen(_when.then(_expr))
  };
}


/**
 * Start a when, then, otherwise expression.
 *
 * @example
 * ```
 * // Below we add a column with the value 1, where column "foo" > 2 and the value -1 where it isn't.
 * >>> df = pl.DataFrame({"foo": [1, 3, 4], "bar": [3, 4, 0]})
 * >>> df.withColumn(pl.when(pl.col("foo").gt(2)).then(pl.lit(1)).otherwise(pl.lit(-1)))
 * shape: (3, 3)
 * ┌─────┬─────┬─────────┐
 * │ foo ┆ bar ┆ literal │
 * │ --- ┆ --- ┆ ---     │
 * │ i64 ┆ i64 ┆ i32     │
 * ╞═════╪═════╪═════════╡
 * │ 1   ┆ 3   ┆ -1      │
 * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
 * │ 3   ┆ 4   ┆ 1       │
 * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
 * │ 4   ┆ 0   ┆ 1       │
 * └─────┴─────┴─────────┘
 *
 * // Or with multiple `when, thens` chained:
 * >>> df.with_column(
 * ...     pl.when(pl.col("foo").gt(2))
 * ...     .then(1)
 * ...     .when(pl.col("bar").gt(2))
 * ...     .then(4)
 * ...     .otherwise(-1)
 * ... )
 * shape: (3, 3)
 * ┌─────┬─────┬─────────┐
 * │ foo ┆ bar ┆ literal │
 * │ --- ┆ --- ┆ ---     │
 * │ i64 ┆ i64 ┆ i32     │
 * ╞═════╪═════╪═════════╡
 * │ 1   ┆ 3   ┆ 4       │
 * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
 * │ 3   ┆ 4   ┆ 1       │
 * ├╌╌╌╌╌┼╌╌╌╌╌┼╌╌╌╌╌╌╌╌╌┤
 * │ 4   ┆ 0   ┆ 1       │
 * └─────┴─────┴─────────┘
 * ```
 */
export function when(expr: Expr): When  {

  return When(pli.when(expr._expr));
}
