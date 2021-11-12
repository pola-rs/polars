import pl from '../polars'

const s = pl.Series('a', [1, 2, 3])
console.log(s)

const s2 = pl.Series('a', [1, 2, 3], {type: "Int8"})
console.log("%O\n%O", "series2",s2)



// Arithmetic
const series = pl.Series({
  name: "foo",
  values: [1, 2, 3]
})

const other = pl.Series({
  name: "foo",
  values: [1, 2, 3]
})
const add = series.add(other)
const sub = series.sub(other)
const mul = series.mul(other)
const div = series.div(other)

console.log("\n>>>>>\n%O\n%O\n<<<<<\n", "series.add()", add)
console.log("\n>>>>>\n%O\n%O\n<<<<<\n", "series.sub()", sub)
console.log("\n>>>>>\n%O\n%O\n<<<<<\n", "series.mul()", mul)
console.log("\n>>>>>\n%O\n%O\n<<<<<\n", "series.div()", div)