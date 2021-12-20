
import * as func from "./lazy_functions";
import * as gb from "./groupby";
import * as expr from "./expr";
import * as whenthen from "./whenthen";

namespace lazy {
  export import GroupBy = gb.LazyGroupBy;
  export import Expr = expr.Expr;
  export import funcs = func
  export import when = whenthen
}

export = lazy;
