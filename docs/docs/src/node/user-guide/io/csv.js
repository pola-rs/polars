// --8<-- [start:setup]
const pl = require("nodejs-polars");
// --8<-- [end:setup]

`
// --8<-- [start:read]
df = pl.readCSV("path.csv")
// --8<-- [end:read]
`;

// --8<-- [start:write]
df = pl.DataFrame({ foo: [1, 2, 3], bar: [null, "bak", "baz"] });
df.writeCSV("path.csv");
// --8<-- [end:write]

// --8<-- [start:scan]
df = pl.scanCSV("path.csv");
// --8<-- [end:scan]
