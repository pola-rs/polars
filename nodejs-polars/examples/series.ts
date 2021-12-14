import pl, {col, lit} from "../polars";
const csvpath = "/home/cgrinstead/Development/git/covalent/blocks.csv";

pl.Config()
  .setTblRows(20)
  .setAsciiTables()
  .setTblWidthChars(140);

const df = pl.readCSV(csvpath);


console.log(df);