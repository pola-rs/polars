import * as dt from "./datetime";
import * as lst from "./list";
import * as str from "./string";
import * as struct from "./struct";

namespace expr {

  export import DateTimeFunctions = dt.ExprDateTimeFunctions;
  export import ListFunctions = lst.ExprListFunctions;
  export import StringFunctions = str.ExprStringFunctions;
  export import StructFunctions = struct.ExprStructFunctions;


  export import List = lst.ExprList
  export import Datetime = dt.ExprDateTime
  export import String = str.ExprString
  export import Struct = struct.ExprStruct
}

export = expr
