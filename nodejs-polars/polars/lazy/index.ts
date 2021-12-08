
import * as func from "./lazy_functions";
import * as gb from "./groupby";
import * as expr from "./expr";


namespace lazy {
  export import GroupBy = gb.LazyGroupBy;
  export import Expr = expr.Expr;
  export import funcs = func

}

export = lazy;
