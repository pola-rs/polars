const pl = require("nodejs-polars");
const Chance = require("chance");
const chance = new Chance();
const R = require("ramda")
const _ = require("lodash")
const fs = require("fs")
const {performance} = require('perf_hooks');

const ints1k = Array.from({length: 1000}, () => chance.integer({min: -1000, max: 1000}));
const strings1k = Array.from({length: 1000}, () => chance.word());
const countries1k = Array.from({length: 1000}, () => chance.country());

const ints1kDS = pl.Series("ints", ints1k);
const strings1kDS = pl.Series("strings", strings1k);
const countries1kDS = pl.Series("countries", countries1k);
const ints10k = Array.from({length: 10000}, () => chance.integer({min: -1000, max: 1000}));
const strings10k = Array.from({length: 10000}, () => chance.word());
const countries10k = Array.from({length: 10000}, () => chance.country());


const ints10kDS = pl.Series("ints", ints10k);
const strings10kDS = pl.Series("strings", strings10k);
const countries10kDS = pl.Series("countries", countries10k);

const ints100k = Array.from({length: 100000}, () => chance.integer({min: -1000, max: 1000}));
const strings100k = Array.from({length: 100000}, () => chance.word());
const countries100k = Array.from({length: 100000}, () => chance.country());

const ints100kDS = pl.Series("ints", ints100k);
const strings100kDS = pl.Series("strings", strings100k);
const countries100kDS = pl.Series("countries", countries100k);
const ints1M = Array.from({length: 1000000}, () => chance.integer({min: -1000, max: 1000}));
const strings1M = Array.from({length: 1000000}, () => chance.word());
const countries1M = Array.from({length: 1000000}, () => chance.country());

const ints1MDS = pl.Series("ints", ints1M);
const strings1MDS = pl.Series("strings", strings1M);
const countries1MDS = pl.Series("countries", countries1M);

const config = {
  numIterations: 100
}

const timeit = (fn) => {

  const iterations = Array.from({length: config.numIterations}).map(() => {
    const t = performance.now()
    fn();
    return performance.now() - t

  })
  const stats = pl.Series(iterations)

  return {
    "mean:ms": Number(stats.mean().toPrecision(3)),
    "min:ms": Number(stats.min().toPrecision(3)),
    "max:ms": Number(stats.max().toPrecision(3)),
    "std": Number(stats.toFrame().std()[0][0].toPrecision(5))
  }
};



const batches = {
  "1k": 1000,
  "10k": 10000,
  "100k": 100000,
  "1M": 1000000
}
const writeStream = fs.createWriteStream("./benchmarks/list-operations.csv", {flags: "a"})
const fmt = (objs) => {

  const entries = Object.entries(objs).flatMap(([batchSize, groups]) => {
    console.log(`running for batch size: ${batchSize}`)
    return Object.entries(groups).flatMap(([data_structure, entries]) => {
      return Object.entries(entries).map(([operation, fn]) => {
        return {
          batchSize,
          batchSizeInt: batches[batchSize],
          operation,
          data_structure,
          ...timeit(fn)
        }
      })
    })
  })
  const df = pl.DataFrame(entries, {orient: "row"})
    .sort({by: ["batchSizeInt", "operation", "mean:ms"]})
    .drop("batchSizeInt")
  df.toCSV(writeStream, {hasHeader: false})
  const table = df
    .rows().map(row => {
      const cols = df.columns
      return row.reduce((acc, curr, idx) => ({...acc, [cols[idx]]: curr}), {})
    })
  console.table(table)
}
fmt(
  {
    "1k": {
      "array": {
        "intsum": () => ints1k.reduce((a, b) => a + b),
        "stringsort": () => strings1k.sort((a, b) => a.localeCompare(b)),
        "valueCounts": () => getValueCounts(ints1k),
        "distinct": () => new Set(strings1k),
        "intsort": () => ints1k.sort((a, b) => a > b),
      },
      "series": {
        "intsort": () => ints1kDS.sort(),
        "intsum": () => ints1kDS.sum(),
        "stringsort": () => strings1kDS.sort(),
        "valueCounts": () => countries1kDS.valueCounts(),
        "distinct": () => strings1kDS.unique(),
      },
      "ramda": {
        "intsum": () => R.sum(ints1k),
        "stringsort": () => R.sort((a, b) => a.localeCompare(b))(strings1k),
        "valueCounts": () => R.countBy(R.identity)(ints1k),
        "distinct": () => R.uniq(strings1k),
        "intsort": () => R.sort((a, b) => a > b, ints1k),
      },
      "lodash": {
        "intsum": () => _.sum(ints1k),
        "stringsort": () => _.sortBy(strings1k, (a, b) => a.localeCompare(b)),
        "valueCounts": () => _.countBy(ints1k, _.identity),
        "distinct": () => _.uniq(strings1k),
        "intsort": () => _.sortBy(ints1k, [(a, b) => a - b]),
      }
    },
    "10k": {
      "array": {
        "intsum": () => ints10k.reduce((a, b) => a + b),
        "stringsort": () => strings10k.sort((a, b) => a.localeCompare(b)),
        "valueCounts": () => getValueCounts(ints10k),
        "distinct": () => new Set(strings10k),
        "intsort": () => ints10k.sort((a, b) => a > b),
      },
      "series": {
        "intsort": () => ints10kDS.sort(),
        "intsum": () => ints10kDS.sum(),
        "stringsort": () => strings10kDS.sort(),
        "valueCounts": () => countries10kDS,
        "distinct": () => strings10kDS.unique(),
      },
      "ramda": {
        "intsum": () => R.sum(ints10k),
        "stringsort": () => R.sort((a, b) => a.localeCompare(b))(strings10k),
        "valueCounts": () => R.countBy(R.identity)(ints10k),
        "distinct": () => R.uniq(strings10k),
        "intsort": () => R.sort((a, b) => a > b, ints10k),
      },
      "lodash": {
        "intsum": () => _.sum(ints10k),
        "stringsort": () => _.sortBy(strings10k, (a, b) => a.localeCompare(b)),
        "valueCounts": () => _.countBy(ints10k, _.identity),
        "distinct": () => _.uniq(strings10k),
        "intsort": () => _.sortBy(ints10k, [(a, b) => a - b])
      }
    },
    "100k": {
      "array": {
        "intsum": () => ints100k.reduce((a, b) => a + b),
        "stringsort": () => strings100k.sort((a, b) => a.localeCompare(b)),
        "valueCounts": () => getValueCounts(ints100k),
        "distinct": () => new Set(strings100k),
        "intsort": () => ints100k.sort((a, b) => a > b),
      },
      "series": {
        "intsort": () => ints100kDS.sort(),
        "stringsort": () => strings100kDS.sort(),
        "intsum": () => ints100kDS.sum(),
        "valueCounts": () => countries100kDS.valueCounts(),
        "distinct": () => strings100kDS.unique(),
      },
      "ramda": {
        "intsum": () => R.sum(ints100k),
        "stringsort": () => R.sort((a, b) => a.localeCompare(b))(strings100k),
        "valueCounts": () => R.countBy(R.identity)(ints100k),
        "distinct": () => R.uniq(strings100k),
        "intsort": () => R.sort((a, b) => a > b, ints100k),
      },
      "lodash": {
        "intsum": () => _.sum(ints100k),
        "stringsort": () => _.sortBy(strings100k, (a, b) => a.localeCompare(b)),
        "valueCounts": () => _.countBy(ints100k, _.identity),
        "distinct": () => _.uniq(strings100k),
        "intsort": () => _.sortBy(ints100k, [(a, b) => a > b])
      }
    },
    "1M": {
      "array": {
        "intsum": () => ints1M.reduce((a, b) => a + b),
        "stringsort": () => strings1M.sort((a, b) => a.localeCompare(b)),
        "valueCounts": () => getValueCounts(ints1M),
        "distinct": () => new Set(strings1M),
        "intsort": () => ints1M.sort((a, b) => a > b),
      },
      "series": {
        "intsort": () => ints1MDS.sort(),
        "stringsort": () => strings1MDS.sort(),
        "intsum": () => ints1MDS.sum(),
        "valueCounts": () => countries1MDS.valueCounts(),
        "distinct": () => strings1MDS.unique(),
      },
      "ramda": {
        "intsum": () => R.sum(ints1M),
        "stringsort": () => R.sort((a, b) => a.localeCompare(b))(strings1M),
        "valueCounts": () => R.countBy(R.identity)(ints1M),
        "distinct": () => R.uniq(strings1M),
        "intsort": () => R.sort((a, b) => a > b, ints1M),
      },
      "lodash": {
        "intsum": () => _.sum(ints1M),
        "stringsort": () => _.sortBy(strings1M, (a, b) => a.localeCompare(b)),
        "valueCounts": () => _.countBy(ints1M, _.identity),
        "distinct": () => _.uniq(strings1M),
        "intsort": () => _.sortBy(ints1M, [(a, b) => a > b])
      }
    },
  }
)


function getValueCounts(values) {
  const acc = {};
  values.forEach(val => {
    if (acc[val]) {
      acc[val] = acc[val] + 1;
    } else {
      acc[val] = 1;
    }
  });

  return acc
}