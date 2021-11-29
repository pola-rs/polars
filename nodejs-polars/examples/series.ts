import pl from "../polars";


const s = pl.Series([[1], [2,2], [0,1]]);
const dt_series = pl.Series("d", [new Date(13241324)], pl.Date);
console.log(dt_series);