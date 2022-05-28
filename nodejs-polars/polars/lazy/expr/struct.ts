import pli from "../../internals/polars_internal";
import {_Expr, Expr} from "../expr";

export interface ExprStruct {
  field(name: string): Expr;
  renameFields(names: string[]): Expr
}


export const ExprStructFunctions = (_expr: any): ExprStruct => {
  return {
    field(name) {
      return _Expr(_expr.structFieldByName(name));
    },
    renameFields(names) {
      return _Expr(_expr.structRenameFields(names));
    }
  };
};
