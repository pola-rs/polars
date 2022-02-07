
import * as func from "./functions";
import * as gb from "./groupby";
import * as expr from "./expr";
import * as whenthen from "./whenthen";
namespace lazy {
  export import GroupBy = gb.LazyGroupBy;
  export import Expr = expr;
  export import funcs = func;
  export import when = whenthen;
}

export = lazy;
