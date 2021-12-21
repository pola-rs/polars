import {Expr, exprToLitOrExpr} from "./expr";
import {Series} from "../series";
import { DataFrame } from "../dataframe";
import { ExprOrString, isSeries, range, selectionToExprList} from "../utils";
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
 * ```
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
export function cols(col1: string, ...cols: string[]): Expr {
  cols.unshift(col1);

  return col(cols);
}

export function lit(value: any): Expr {
  if(Array.isArray(value)) {
    value = Series(value);
  }
  if(isSeries(value)){

    return Expr(pli.lit({value: value._series}));
  }

  return Expr(pli.lit({value}));
}

// ----------
// Helper Functions
// ------


/**
 * Get the maximum value. Can be used horizontally or vertically.
 * @param column
 */
// export function max(column: ColumnsOrExpr): Expr;
// export function max<T>(column: Series<T>): number | bigint;
// export function max(column) {}


// export function fold<T>(
//   acc: Expr,
//   func: (curr:Series<T>, next: Series<T>) => Series<T>,
//   exprs: ColumnsOrExpr
// ): Expr {
//   exprs = [exprs]
//     .flat(3)
//     .map(e => exprToLitOrExpr(e, false)._expr);

//   return null as any;
// }

/**
 * __Create a range expression.__
 * ___
 *
 * This can be used in a `select`, `with_column` etc.
 * Be sure that the range size is equal to the DataFrame you are collecting.
 * @param low - Lower bound of range.
 * @param high - Upper bound of range.
 * @param step - Step size of the range
 * @param eager - If eager evaluation is `true`, a Series is returned instead of an Expr
 * @example
 * ```
 * >>> df.lazy()
 * >>>   .filter(pl.col("foo").lt(pl.arange(0, 100)))
 * >>>   .collect()
 * ```
 */
export function arange<T>(opts: {low: Series<T> | T, high: Series<T> | T, step: number, eager: boolean});
export function arange<T>(low: Series<T> | T, high?: Series<T> | T, step?: number, eager?: true): Series<T>;
export function arange<T>(low: Series<T> | T, high?: Series<T> | T, step?: number, eager?: false): Expr;
export function arange<T>(opts: any, high?, step?, eager?): Series<T> | Expr {
  if(typeof opts?.low === "number") {
    return arange(opts.low, opts.high, opts.step, opts.eager);
  } else {
    const low = exprToLitOrExpr(opts, false);
    high = exprToLitOrExpr(high, false);
    if(eager) {
      const df = DataFrame({"a": [1]});

      return df.select(arange(low, high, step).alias("arange") as any).getColumn("arange") as any;
    }

    return Expr(pli.arange({low: low._expr, high: high._expr, step}));
  }
}
/**
 * __Find the indexes that would sort the columns.__
 * ___
 * Argsort by multiple columns. The first column will be used for the ordering.
 * If there are duplicates in the first column, the second column will be used to determine the ordering
 * and so on.
 */

export function argSortBy(exprs: Expr[] | string[], reverse: boolean | boolean[] = false): Expr {
  if(!Array.isArray(reverse)) {
    reverse = Array.from({length: exprs.length}, () => reverse) as any;
  }
  const by = selectionToExprList(exprs);

  return Expr(pli.argSortBy({by, reverse}));
}
/** Alias for mean. @see {@link mean} */
export function avg(column: string): Expr;
export function avg(column: Series<any>): number;
export function avg(column: Series<any> | string): number | Expr {
  return mean(column as any);
}

/**
 * Concat the arrays in a Series dtype List in linear time.
 * @param exprs Columns to concat into a List Series
 */
export function concatList(exprs: ExprOrString[]): Expr
export function concatList(expr: ExprOrString, ...exprs: ExprOrString[]): Expr
export function concatList(expr: ExprOrString, expr2: ExprOrString,  ...exprs: ExprOrString[]): Expr
export function concatList(...exprs): Expr {
  const items = selectionToExprList(exprs as any, false);

  return Expr(pli.concatList({items}));
}

/** Concat Utf8 Series in linear time. Non utf8 columns are cast to utf8. */
export function concatString(opts: {exprs: ExprOrString[], sep: string});
export function concatString(exprs: ExprOrString[], sep?: string);
export function concatString(opts, sep=",") {
  if(opts?.exprs) {
    return concatString(opts.exprs, opts.sep);
  }
  const items = selectionToExprList(opts as any, false);

  return Expr(pli.concatString({items, sep}));

}

/** Count the number of values in this column. */
export function count(column: string): Expr
export function count(column: Series<any>): number
export function count(column: string | Series<any>): Expr | number {
  if(isSeries(column)) {
    return column.len();
  } else {
    return col(column).count();
  }
}

/** Compute the covariance between two columns/ expressions. */
export function cov(a: ExprOrString, b: ExprOrString): Expr {
  a = exprToLitOrExpr(a, false)._expr;
  b = exprToLitOrExpr(b, false)._expr;

  return Expr(pli.cov({a, b}));
}
/**
 * Exclude certain columns from a wildcard expression.
 *
 * Syntactic sugar for:
 * ```
 * >>> pl.col("*").exclude(columns)
 * ```
 */
export function exclude(column: string, ...columns: string[]) {
  return col("*").exclude(column, ...columns);
}

/** Get the first value. */
export function first(column: string): Expr
export function first<T>(column: Series<T>): T
export function first<T>(column: string | Series<T>): Expr | T {
  if(isSeries(column)) {
    if(column.length) {
      return column[0];
    } else {
      throw new RangeError("The series is empty, so no first value can be returned.");
    }
  } else {
    return col(column).first();
  }
}

/**
 * String format utility for expressions
 * Note: strings will be interpolated as `col(<value>)`. if you want a literal string, use `lit(<value>)`
 * @example
 * ```
 * >>> df = pl.DataFrame({
 * ...   "a": ["a", "b", "c"],
 * ...   "b": [1, 2, 3],
 * ... })
 * >>> df.select(
 * ...   pl.format("foo_{}_bar_{}", pl.col("a"), "b").alias("fmt"),
 * ... )
 * shape: (3, 1)
 * ┌─────────────┐
 * │ fmt         │
 * │ ---         │
 * │ str         │
 * ╞═════════════╡
 * │ foo_a_bar_1 │
 * ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
 * │ foo_b_bar_2 │
 * ├╌╌╌╌╌╌╌╌╌╌╌╌╌┤
 * │ foo_c_bar_3 │
 * └─────────────┘
 *
 * // You can use format as tag function as well
 * >>> pl.format("foo_{}_bar_{}", pl.col("a"), "b") // is the same as
 * >>> pl.format`foo_${pl.col("a")}_bar_${"b"}`
 * ```
 */
export function format(fstring: string, ...expr: ExprOrString[]): Expr
export function format(strings: TemplateStringsArray, ...expr: ExprOrString[]): Expr
export function format(strings, ...expr): Expr {
  if(typeof strings === "string") {
    const s = strings.split("{}");
    if(s.length - 1 !== expr.length) {
      throw new RangeError("number of placeholders should equal the number of arguments");
    }

    return format(s as any, ...expr);
  }
  const d = range(0, Math.max(strings.length, expr.length)).flatMap((i) => {
    const sVal = strings[i] ? lit(strings[i]) : [];
    const exprVal = expr[i] ? exprToLitOrExpr(expr[i], false) : [];

    return [sVal, exprVal];
  })
    .flat();

  return concatString(d, "");
}
/** Syntactic sugar for `pl.col(column).aggGroups()` */
export function groups(column: string): Expr {
  return col(column).aggGroups();
}
/** Get the first n rows of an Expression. */
export function head(column: ExprOrString, n?: number): Expr;
export function head<T>(column: Series<T>, n?: number): Series<T>;
export function head<T>(column: Series<T> | ExprOrString, n?): Series<T> | Expr {
  if(isSeries(column)) {
    return column.head(n);
  } else {
    return exprToLitOrExpr(column, false).head(n);

  }
}

/** Get the last value. */
export function last(column: ExprOrString): Expr
export function last<T>(column: Series<T>): T
export function last<T>(column: ExprOrString | Series<T>): Expr | T {
  if(isSeries(column)) {
    if(column.length) {
      return column[-1];
    } else {
      throw new RangeError("The series is empty, so no last value can be returned.");
    }
  } else {
    return exprToLitOrExpr(column, false).last();
  }
}

/** Get the mean value. */
export function mean(column: ExprOrString): Expr;
export function mean(column: Series<any>): number;
export function mean(column: Series<any> | ExprOrString): number | Expr {
  if(isSeries(column)) {
    return column.mean();
  }

  return exprToLitOrExpr(column, false).mean();
}

/** Get the median value. */
export function median(column: ExprOrString): Expr;
export function median(column: Series<any>): number;
export function median(column: Series<any> | ExprOrString): number | Expr {
  if(isSeries(column)) {
    return column.median();
  }

  return exprToLitOrExpr(column, false).median();
}

/** Count unique values. */
export function nUnique(column: ExprOrString): Expr;
export function nUnique(column: Series<any>): number;
export function nUnique(column: Series<any> | ExprOrString): number | Expr {
  if(isSeries(column)) {
    return column.nUnique();
  }

  return exprToLitOrExpr(column, false).nUnique();
}


/** Compute the pearson's correlation between two columns. */
export function pearsonCorr(a: ExprOrString, b: ExprOrString): Expr {
  a = exprToLitOrExpr(a, false)._expr;
  b = exprToLitOrExpr(b, false)._expr;

  return Expr(pli.pearsonCorr({a, b}));
}

/** Get the quantile */
export function quantile(column: ExprOrString, q: number): Expr;
export function quantile(column: Series<any>, q: number): number;
export function quantile(column, q) {
  if(isSeries(column)) {
    return column.quantile(q);
  }

  return exprToLitOrExpr(column, false).quantile(q);
}
/**
 * __Run polars expressions without a context.__
 *
 * This is syntactic sugar for running `df.select` on an empty DataFrame.
 */
export function select(expr: ExprOrString, ...exprs: ExprOrString[]) {
  return DataFrame({}).select(expr, ...exprs);
}

/** Compute the spearman rank correlation between two columns. */
export function spearmanRankCorr(a: ExprOrString, b: ExprOrString): Expr {
  a = exprToLitOrExpr(a, false)._expr;
  b = exprToLitOrExpr(b, false)._expr;

  return Expr(pli.spearmanRankCorr({a, b}));
}


/** Get the last n rows of an Expression. */
export function tail(column: ExprOrString, n?: number): Expr;
export function tail<T>(column: Series<T>, n?: number): Series<T>;
export function tail<T>(column: Series<T> | ExprOrString, n?: number): Series<T> | Expr {
  if(isSeries(column)) {
    return column.tail(n);
  } else {
    return exprToLitOrExpr(column, false).tail(n);
  }
}

/** Syntactic sugar for `pl.col(column).list()` */
export function list(column: ExprOrString): Expr {
  return exprToLitOrExpr(column, false).list();
}


// export function collect_all() {}
// export function all() {} // fold
// export function any() {} // fold
// export function apply() {} // lambda
// export function fold() {}
// export function map_binary() {} //lambda
// export function map() {} //lambda
// export function max() {} // fold
// export function min() {} // fold
// export function sum() {} // fold
