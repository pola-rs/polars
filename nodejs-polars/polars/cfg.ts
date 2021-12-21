export interface Config {
  /** Use utf8 characters to print tables */
  setUtf8Tables(): Config
  /** Use ascii characters to print tables */
  setAsciiTables(): Config
  /** Set the number of character used to draw the table */
  setTblWidthChars(width: number): Config
  /** Set the number of rows used to print tables */
  setTblRows(n: number): Config
  /** Set the number of columns used to print tables */
  setTblCols(n: number): Config
  /** Turn on the global string cache */
  setGlobalStringCache(): Config
  /** Turn off the global string cache */
  unsetGlobalStringCache(): Config
}
export const Config = (): Config => {
  return {
    setUtf8Tables() {
      process.env["POLARS_FMT_NO_UTF8"] = undefined;

      return this;
    },
    setAsciiTables() {
      process.env["POLARS_FMT_NO_UTF8"] = "1";

      return this;
    },
    setTblWidthChars(width) {
      process.env["POLARS_TABLE_WIDTH"] = String(width);

      return this;
    },
    setTblRows(n) {
      process.env["POLARS_FMT_MAX_ROWS"] = String(n);

      return this;
    },
    setTblCols(n) {
      process.env["POLARS_FMT_MAX_COLS"] = String(n);

      return this;
    },
    setGlobalStringCache() {
      return this;
    },
    unsetGlobalStringCache() {
      return this;
    }

  };
};
