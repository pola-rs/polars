// import pl_rs from "./polars_internal";
// import util from "util"
// import pl from './'
// /**
//  * 
//  * @description
//  * A DataFrame is a two-dimensional data structure that represents data as a table 
//  * with rows and columns.
//  */
// export default class Dataframe {
//   private df: Dataframe;
//   static from = (objects: Record<string, any[]>) => {
//     const columns = Object.entries(objects)
//       .map(([key, values]: any) => pl.Series(key, values)._series);
//     return new Dataframe(pl_rs.df.new_obj({columns}));
//   }
//   static fromRows = (js_objects: Array<Record<any, any>>) => {

//     return new Dataframe(pl_rs.df.from_rows({js_objects}));
//   }
//   constructor(df: Dataframe) {
//     this.df = df
//   }
//   private of(method: string, args: Object) {
//     return new Dataframe(pl_rs.df[method]({_df: this.df, ...args}))
//   }
//   private unwrap(method: string) {
//     return pl_rs.df[method]({_df: this.df})
//   }

//   head = (length = 5) => this.of('head', {length});
//   height = () => this.unwrap('height');
//   isEmpty = () => this.unwrap('is_empty');
//   shape = () => this.unwrap('shape');
//   width = () => this.unwrap('width');

//   [util.inspect.custom]() {

//     return this.unwrap('get_fmt')
//   }
// }



