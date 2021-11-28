/* eslint-disable no-undef */
import fs from 'fs';
import pl from '../polars';
import path from "path";
import {Stream} from 'stream';

const jsonpath = "/home/cgrinstead/Development/git/covalent/blocks.json";
const csvpath = "/home/cgrinstead/Development/git/covalent/blocks.csv";


const data = [
  ['11','33','pp','123','123','123',],
  ['cc','aa','bb','rr','ss','tt',],
];
const csv = fs.readFileSync(csvpath, "utf-8");

const df = pl.readCSV(csvpath);
const df2 = pl.readCSV(df.toCSV());

console.log(df.head());
console.log(df.frameEqual(df2));