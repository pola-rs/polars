/* eslint-disable no-undef */
import fs from 'fs';
import pl from '../polars';
import path from "path";
import {Stream} from 'stream';

import { Readable } from 'stream';


// const data = [
//   {"hash": "0xb7f9f1f6ec6fcddbace65662c1076f7333fdfeb953adfb44775ecb197fbae5a5",}
//   ,{"hash": "0x967491a63679cc0bc6a6199ada31a15eaf659a2eb75329672b8f3365da928ec0"}
//   ,{"hash": "0xec7e32cba4affc8dc3e74006e6d53af04949d7d2b4c5b00263c5432ed120d47b"}
//   ,{"hash": "0x0f667485d6315f94321603b8f6bf77d1378b0ef639c8c4339fd2a27e22681c64"}
//   ,{"hash": "0xe16d977c0a8c1f36f8872c94cac3a2e9f3cb3c1de4c139ce15fac178b1aea8fd"}
//   ,{"hash": "0x533af700a9583e73936dcac4f0d8038e0a7fbd06df13984879b8517fdd8f7980"}
//   ,{"hash": "0xd352d9a0d4478c1905fb6c368ec4d4ef7ae4beea7bbdc2dc6ff350accb928c0b"}
//   ,{"hash": "0x5ea6e63282685f2866e7316be3619f898e5585042e9d018917a01bc4d6488e5b"}
//   ,{"hash": "0x4a97e2f8c172cfa7e2d48fe0931a9faf16482de4e469afe420e5a54dce7b3ed2"}
//   ,{"hash": "0xb8128fe000a06a18d0bcaea7643231dda0b30d7a4e6a6f649ecd906976b5c78a"}
//   ,
// ];
// const dataReadable = Readable.from(data);

// async function read(readable) {
//   readable.setEncoding('utf8');
//   let data = '';

//   for await (const chunk of readable) {
//     data += chunk;
//   }

//   return data;
// }

// console.log(read(dataReadable));


const jsonpath = "/home/cgrinstead/Development/git/ethereum-etl/transactions.json";
const csvpath = "/home/cgrinstead/Development/git/ethereum-etl/polyout/transactions/start_block=16082460/end_block=16082465/transactions_16082460_16082465.csv";


const jsonStream = fs.createReadStream(jsonpath);


// console.log(Object.getOwnPropertyNames(jsonStream));
// console.log(jsonStream);

// const d = pl._internal.df.read_json({
//   path: jsonFile,
//   inline: true,
//   infer_schema_length: 10,
//   batch_size: 10
// });

const csvdf = pl._internal.df.read_csv({
  path: csvpath,
  sep: ",",
  chunk_size: 3,
  has_header: true,
  rechunk: false,
  encoding: "utf8",
  parse_dates: false,
  low_memory: false,
  ignore_errors: true,
  skip_rows: 0
});

const data = {
  "a": [1,2,3,4],
  "b": [true, true, null, false]
};

function obj_to_df(obj: Record<string, Array<any>>, columns?: Array<string>): any {
  const data =  Object.entries(obj).map(([key, value], idx) => {
    return pl.Series(columns?.[idx] ?? key, value)._series;
  });
  

  return pl._internal.df.new_from_columns({columns: data});
}


const rows = [
  [1,2,3],
  [2,3,4]
];
// Buffer.from(rows)

const d = pl._internal.df.read_json({inline: true, batch_size: 3, path: rows});


// let s = pl._internal.df.select({_df: d, selection: ["hash", "nonce"]});
let s = pl._internal.df.as_str({_df: d});


// // console.log({df: d, str: s});
console.log(s);