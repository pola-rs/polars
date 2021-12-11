import pl from "../polars";
import pli from "../polars/internals/polars_internal";
const w = pli
  .when(pl.col("foo"));
console.log(w);

// function whenthen(predicate, arg) {
// }

// function when(predicate) {
//   return {
//     then: (expr)  => {
//       console.log({predicate, _this: this});

//       return whenthen(predicate, expr);
//     }
//   };
// }
// when("foo").then("bar");