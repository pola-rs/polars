const { existsSync, readFileSync } = require('fs')
const { join } = require('path')

const { platform, arch } = process

let nativeBinding = null
let localFileExisted = false
let loadError = null

function isMusl() {
  // For Node 10
  if (!process.report || typeof process.report.getReport !== 'function') {
    try {
      return readFileSync('/usr/bin/ldd', 'utf8').includes('musl')
    } catch (e) {
      return true
    }
  } else {
    const { glibcVersionRuntime } = process.report.getReport().header
    return !glibcVersionRuntime
  }
}

switch (platform) {
  case 'android':
    switch (arch) {
      case 'arm64':
        localFileExisted = existsSync(join(__dirname, 'nodejs-polars.android-arm64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.android-arm64.node')
          } else {
            nativeBinding = require('nodejs-polars-android-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm':
        localFileExisted = existsSync(join(__dirname, 'nodejs-polars.android-arm-eabi.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.android-arm-eabi.node')
          } else {
            nativeBinding = require('nodejs-polars-android-arm-eabi')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Android ${arch}`)
    }
    break
  case 'win32':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(
          join(__dirname, 'nodejs-polars.win32-x64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.win32-x64-msvc.node')
          } else {
            nativeBinding = require('nodejs-polars-win32-x64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'ia32':
        localFileExisted = existsSync(
          join(__dirname, 'nodejs-polars.win32-ia32-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.win32-ia32-msvc.node')
          } else {
            nativeBinding = require('nodejs-polars-win32-ia32-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'nodejs-polars.win32-arm64-msvc.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.win32-arm64-msvc.node')
          } else {
            nativeBinding = require('nodejs-polars-win32-arm64-msvc')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Windows: ${arch}`)
    }
    break
  case 'darwin':
    switch (arch) {
      case 'x64':
        localFileExisted = existsSync(join(__dirname, 'nodejs-polars.darwin-x64.node'))
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.darwin-x64.node')
          } else {
            nativeBinding = require('nodejs-polars-darwin-x64')
          }
        } catch (e) {
          loadError = e
        }
        break
      case 'arm64':
        localFileExisted = existsSync(
          join(__dirname, 'nodejs-polars.darwin-arm64.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.darwin-arm64.node')
          } else {
            nativeBinding = require('nodejs-polars-darwin-arm64')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on macOS: ${arch}`)
    }
    break
  case 'freebsd':
    if (arch !== 'x64') {
      throw new Error(`Unsupported architecture on FreeBSD: ${arch}`)
    }
    localFileExisted = existsSync(join(__dirname, 'nodejs-polars.freebsd-x64.node'))
    try {
      if (localFileExisted) {
        nativeBinding = require('./nodejs-polars.freebsd-x64.node')
      } else {
        nativeBinding = require('nodejs-polars-freebsd-x64')
      }
    } catch (e) {
      loadError = e
    }
    break
  case 'linux':
    switch (arch) {
      case 'x64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'nodejs-polars.linux-x64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./nodejs-polars.linux-x64-musl.node')
            } else {
              nativeBinding = require('nodejs-polars-linux-x64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'nodejs-polars.linux-x64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./nodejs-polars.linux-x64-gnu.node')
            } else {
              nativeBinding = require('nodejs-polars-linux-x64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm64':
        if (isMusl()) {
          localFileExisted = existsSync(
            join(__dirname, 'nodejs-polars.linux-arm64-musl.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./nodejs-polars.linux-arm64-musl.node')
            } else {
              nativeBinding = require('nodejs-polars-linux-arm64-musl')
            }
          } catch (e) {
            loadError = e
          }
        } else {
          localFileExisted = existsSync(
            join(__dirname, 'nodejs-polars.linux-arm64-gnu.node')
          )
          try {
            if (localFileExisted) {
              nativeBinding = require('./nodejs-polars.linux-arm64-gnu.node')
            } else {
              nativeBinding = require('nodejs-polars-linux-arm64-gnu')
            }
          } catch (e) {
            loadError = e
          }
        }
        break
      case 'arm':
        localFileExisted = existsSync(
          join(__dirname, 'nodejs-polars.linux-arm-gnueabihf.node')
        )
        try {
          if (localFileExisted) {
            nativeBinding = require('./nodejs-polars.linux-arm-gnueabihf.node')
          } else {
            nativeBinding = require('nodejs-polars-linux-arm-gnueabihf')
          }
        } catch (e) {
          loadError = e
        }
        break
      default:
        throw new Error(`Unsupported architecture on Linux: ${arch}`)
    }
    break
  default:
    throw new Error(`Unsupported OS: ${platform}, architecture: ${arch}`)
}

if (!nativeBinding) {
  if (loadError) {
    throw loadError
  }
  throw new Error(`Failed to load native binding`)
}

const { version, toggleStringCache, JsDataFrame, readCsv, readJson, readParquet, readIpc, readAvro, fromRows, DataType, horizontalConcat, JsLazyGroupBy, JsLazyFrame, scanCsv, scanParquet, scanIpc, JsExpr, When, WhenThen, WhenThenThen, when, col, count, first, last, cols, dtypeCols, arange, pearsonCorr, spearmanRankCorr, cov, argsortBy, lit, range, concatLst, concatStr, JsSeries, seriesSetAtIdxStr, seriesSetAtIdxF64, seriesSetAtIdxF32, seriesSetAtIdxU8, seriesSetAtIdxU16, seriesSetAtIdxU32, seriesSetAtIdxU64, seriesSetAtIdxI8, seriesSetAtIdxI16, seriesSetAtIdxI32, seriesSetAtIdxI64, seriesSetWithMaskStr, seriesSetWithMaskF64, seriesSetWithMaskF32, seriesSetWithMaskU8, seriesSetWithMaskU16, seriesSetWithMaskU32, seriesSetWithMaskU64, seriesSetWithMaskI8, seriesSetWithMaskI16, seriesSetWithMaskI32, seriesSetWithMaskI64, seriesGetF32, seriesGetF64, seriesGetU8, seriesGetU16, seriesGetU32, seriesGetU64, seriesGetI8, seriesGetI16, seriesGetI32, seriesGetI64, seriesGetDate, seriesGetDatetime, seriesGetDuration, seriesGetStr, seriesAddU8, seriesAddU16, seriesAddU32, seriesAddU64, seriesAddI8, seriesAddI16, seriesAddI32, seriesAddI64, seriesAddDatetime, seriesAddDuration, seriesAddF32, seriesAddF64, seriesSubU8, seriesSubU16, seriesSubU32, seriesSubU64, seriesSubI8, seriesSubI16, seriesSubI32, seriesSubI64, seriesSubDatetime, seriesSubDuration, seriesSubF32, seriesSubF64, seriesDivU8, seriesDivU16, seriesDivU32, seriesDivU64, seriesDivI8, seriesDivI16, seriesDivI32, seriesDivI64, seriesDivF32, seriesDivF64, seriesMulU8, seriesMulU16, seriesMulU32, seriesMulU64, seriesMulI8, seriesMulI16, seriesMulI32, seriesMulI64, seriesMulF32, seriesMulF64, seriesRemU8, seriesRemU16, seriesRemU32, seriesRemU64, seriesRemI8, seriesRemI16, seriesRemI32, seriesRemI64, seriesRemF32, seriesRemF64, seriesAddU8Rhs, seriesAddU16Rhs, seriesAddU32Rhs, seriesAddU64Rhs, seriesAddI8Rhs, seriesAddI16Rhs, seriesAddI32Rhs, seriesAddI64Rhs, seriesAddF32Rhs, seriesAddF64Rhs, seriesSubU8Rhs, seriesSubU16Rhs, seriesSubU32Rhs, seriesSubU64Rhs, seriesSubI8Rhs, seriesSubI16Rhs, seriesSubI32Rhs, seriesSubI64Rhs, seriesSubF32Rhs, seriesSubF64Rhs, seriesDivU8Rhs, seriesDivU16Rhs, seriesDivU32Rhs, seriesDivU64Rhs, seriesDivI8Rhs, seriesDivI16Rhs, seriesDivI32Rhs, seriesDivI64Rhs, seriesDivF32Rhs, seriesDivF64Rhs, seriesMulU8Rhs, seriesMulU16Rhs, seriesMulU32Rhs, seriesMulU64Rhs, seriesMulI8Rhs, seriesMulI16Rhs, seriesMulI32Rhs, seriesMulI64Rhs, seriesMulF32Rhs, seriesMulF64Rhs, seriesRemU8Rhs, seriesRemU16Rhs, seriesRemU32Rhs, seriesRemU64Rhs, seriesRemI8Rhs, seriesRemI16Rhs, seriesRemI32Rhs, seriesRemI64Rhs, seriesRemF32Rhs, seriesRemF64Rhs, seriesEqU8, seriesEqU16, seriesEqU32, seriesEqU64, seriesEqI8, seriesEqI16, seriesEqI32, seriesEqI64, seriesEqF32, seriesEqF64, seriesEqStr, seriesNeqU8, seriesNeqU16, seriesNeqU32, seriesNeqU64, seriesNeqI8, seriesNeqI16, seriesNeqI32, seriesNeqI64, seriesNeqF32, seriesNeqF64, seriesNeqStr, seriesGtU8, seriesGtU16, seriesGtU32, seriesGtU64, seriesGtI8, seriesGtI16, seriesGtI32, seriesGtI64, seriesGtF32, seriesGtF64, seriesGtStr, seriesGtEqU8, seriesGtEqU16, seriesGtEqU32, seriesGtEqU64, seriesGtEqI8, seriesGtEqI16, seriesGtEqI32, seriesGtEqI64, seriesGtEqF32, seriesGtEqF64, seriesGtEqStr, seriesLtU8, seriesLtU16, seriesLtU32, seriesLtU64, seriesLtI8, seriesLtI16, seriesLtI32, seriesLtI64, seriesLtF32, seriesLtF64, seriesLtStr, seriesLtEqU8, seriesLtEqU16, seriesLtEqU32, seriesLtEqU64, seriesLtEqI8, seriesLtEqI16, seriesLtEqI32, seriesLtEqI64, seriesLtEqF32, seriesLtEqF64, seriesLtEqStr } = nativeBinding

module.exports.version = version
module.exports.toggleStringCache = toggleStringCache
module.exports.JsDataFrame = JsDataFrame
module.exports.readCsv = readCsv
module.exports.readJson = readJson
module.exports.readParquet = readParquet
module.exports.readIpc = readIpc
module.exports.readAvro = readAvro
module.exports.fromRows = fromRows
module.exports.DataType = DataType
module.exports.horizontalConcat = horizontalConcat
module.exports.JsLazyGroupBy = JsLazyGroupBy
module.exports.JsLazyFrame = JsLazyFrame
module.exports.scanCsv = scanCsv
module.exports.scanParquet = scanParquet
module.exports.scanIpc = scanIpc
module.exports.JsExpr = JsExpr
module.exports.When = When
module.exports.WhenThen = WhenThen
module.exports.WhenThenThen = WhenThenThen
module.exports.when = when
module.exports.col = col
module.exports.count = count
module.exports.first = first
module.exports.last = last
module.exports.cols = cols
module.exports.dtypeCols = dtypeCols
module.exports.arange = arange
module.exports.pearsonCorr = pearsonCorr
module.exports.spearmanRankCorr = spearmanRankCorr
module.exports.cov = cov
module.exports.argsortBy = argsortBy
module.exports.lit = lit
module.exports.range = range
module.exports.concatLst = concatLst
module.exports.concatStr = concatStr
module.exports.JsSeries = JsSeries
module.exports.seriesSetAtIdxStr = seriesSetAtIdxStr
module.exports.seriesSetAtIdxF64 = seriesSetAtIdxF64
module.exports.seriesSetAtIdxF32 = seriesSetAtIdxF32
module.exports.seriesSetAtIdxU8 = seriesSetAtIdxU8
module.exports.seriesSetAtIdxU16 = seriesSetAtIdxU16
module.exports.seriesSetAtIdxU32 = seriesSetAtIdxU32
module.exports.seriesSetAtIdxU64 = seriesSetAtIdxU64
module.exports.seriesSetAtIdxI8 = seriesSetAtIdxI8
module.exports.seriesSetAtIdxI16 = seriesSetAtIdxI16
module.exports.seriesSetAtIdxI32 = seriesSetAtIdxI32
module.exports.seriesSetAtIdxI64 = seriesSetAtIdxI64
module.exports.seriesSetWithMaskStr = seriesSetWithMaskStr
module.exports.seriesSetWithMaskF64 = seriesSetWithMaskF64
module.exports.seriesSetWithMaskF32 = seriesSetWithMaskF32
module.exports.seriesSetWithMaskU8 = seriesSetWithMaskU8
module.exports.seriesSetWithMaskU16 = seriesSetWithMaskU16
module.exports.seriesSetWithMaskU32 = seriesSetWithMaskU32
module.exports.seriesSetWithMaskU64 = seriesSetWithMaskU64
module.exports.seriesSetWithMaskI8 = seriesSetWithMaskI8
module.exports.seriesSetWithMaskI16 = seriesSetWithMaskI16
module.exports.seriesSetWithMaskI32 = seriesSetWithMaskI32
module.exports.seriesSetWithMaskI64 = seriesSetWithMaskI64
module.exports.seriesGetF32 = seriesGetF32
module.exports.seriesGetF64 = seriesGetF64
module.exports.seriesGetU8 = seriesGetU8
module.exports.seriesGetU16 = seriesGetU16
module.exports.seriesGetU32 = seriesGetU32
module.exports.seriesGetU64 = seriesGetU64
module.exports.seriesGetI8 = seriesGetI8
module.exports.seriesGetI16 = seriesGetI16
module.exports.seriesGetI32 = seriesGetI32
module.exports.seriesGetI64 = seriesGetI64
module.exports.seriesGetDate = seriesGetDate
module.exports.seriesGetDatetime = seriesGetDatetime
module.exports.seriesGetDuration = seriesGetDuration
module.exports.seriesGetStr = seriesGetStr
module.exports.seriesAddU8 = seriesAddU8
module.exports.seriesAddU16 = seriesAddU16
module.exports.seriesAddU32 = seriesAddU32
module.exports.seriesAddU64 = seriesAddU64
module.exports.seriesAddI8 = seriesAddI8
module.exports.seriesAddI16 = seriesAddI16
module.exports.seriesAddI32 = seriesAddI32
module.exports.seriesAddI64 = seriesAddI64
module.exports.seriesAddDatetime = seriesAddDatetime
module.exports.seriesAddDuration = seriesAddDuration
module.exports.seriesAddF32 = seriesAddF32
module.exports.seriesAddF64 = seriesAddF64
module.exports.seriesSubU8 = seriesSubU8
module.exports.seriesSubU16 = seriesSubU16
module.exports.seriesSubU32 = seriesSubU32
module.exports.seriesSubU64 = seriesSubU64
module.exports.seriesSubI8 = seriesSubI8
module.exports.seriesSubI16 = seriesSubI16
module.exports.seriesSubI32 = seriesSubI32
module.exports.seriesSubI64 = seriesSubI64
module.exports.seriesSubDatetime = seriesSubDatetime
module.exports.seriesSubDuration = seriesSubDuration
module.exports.seriesSubF32 = seriesSubF32
module.exports.seriesSubF64 = seriesSubF64
module.exports.seriesDivU8 = seriesDivU8
module.exports.seriesDivU16 = seriesDivU16
module.exports.seriesDivU32 = seriesDivU32
module.exports.seriesDivU64 = seriesDivU64
module.exports.seriesDivI8 = seriesDivI8
module.exports.seriesDivI16 = seriesDivI16
module.exports.seriesDivI32 = seriesDivI32
module.exports.seriesDivI64 = seriesDivI64
module.exports.seriesDivF32 = seriesDivF32
module.exports.seriesDivF64 = seriesDivF64
module.exports.seriesMulU8 = seriesMulU8
module.exports.seriesMulU16 = seriesMulU16
module.exports.seriesMulU32 = seriesMulU32
module.exports.seriesMulU64 = seriesMulU64
module.exports.seriesMulI8 = seriesMulI8
module.exports.seriesMulI16 = seriesMulI16
module.exports.seriesMulI32 = seriesMulI32
module.exports.seriesMulI64 = seriesMulI64
module.exports.seriesMulF32 = seriesMulF32
module.exports.seriesMulF64 = seriesMulF64
module.exports.seriesRemU8 = seriesRemU8
module.exports.seriesRemU16 = seriesRemU16
module.exports.seriesRemU32 = seriesRemU32
module.exports.seriesRemU64 = seriesRemU64
module.exports.seriesRemI8 = seriesRemI8
module.exports.seriesRemI16 = seriesRemI16
module.exports.seriesRemI32 = seriesRemI32
module.exports.seriesRemI64 = seriesRemI64
module.exports.seriesRemF32 = seriesRemF32
module.exports.seriesRemF64 = seriesRemF64
module.exports.seriesAddU8Rhs = seriesAddU8Rhs
module.exports.seriesAddU16Rhs = seriesAddU16Rhs
module.exports.seriesAddU32Rhs = seriesAddU32Rhs
module.exports.seriesAddU64Rhs = seriesAddU64Rhs
module.exports.seriesAddI8Rhs = seriesAddI8Rhs
module.exports.seriesAddI16Rhs = seriesAddI16Rhs
module.exports.seriesAddI32Rhs = seriesAddI32Rhs
module.exports.seriesAddI64Rhs = seriesAddI64Rhs
module.exports.seriesAddF32Rhs = seriesAddF32Rhs
module.exports.seriesAddF64Rhs = seriesAddF64Rhs
module.exports.seriesSubU8Rhs = seriesSubU8Rhs
module.exports.seriesSubU16Rhs = seriesSubU16Rhs
module.exports.seriesSubU32Rhs = seriesSubU32Rhs
module.exports.seriesSubU64Rhs = seriesSubU64Rhs
module.exports.seriesSubI8Rhs = seriesSubI8Rhs
module.exports.seriesSubI16Rhs = seriesSubI16Rhs
module.exports.seriesSubI32Rhs = seriesSubI32Rhs
module.exports.seriesSubI64Rhs = seriesSubI64Rhs
module.exports.seriesSubF32Rhs = seriesSubF32Rhs
module.exports.seriesSubF64Rhs = seriesSubF64Rhs
module.exports.seriesDivU8Rhs = seriesDivU8Rhs
module.exports.seriesDivU16Rhs = seriesDivU16Rhs
module.exports.seriesDivU32Rhs = seriesDivU32Rhs
module.exports.seriesDivU64Rhs = seriesDivU64Rhs
module.exports.seriesDivI8Rhs = seriesDivI8Rhs
module.exports.seriesDivI16Rhs = seriesDivI16Rhs
module.exports.seriesDivI32Rhs = seriesDivI32Rhs
module.exports.seriesDivI64Rhs = seriesDivI64Rhs
module.exports.seriesDivF32Rhs = seriesDivF32Rhs
module.exports.seriesDivF64Rhs = seriesDivF64Rhs
module.exports.seriesMulU8Rhs = seriesMulU8Rhs
module.exports.seriesMulU16Rhs = seriesMulU16Rhs
module.exports.seriesMulU32Rhs = seriesMulU32Rhs
module.exports.seriesMulU64Rhs = seriesMulU64Rhs
module.exports.seriesMulI8Rhs = seriesMulI8Rhs
module.exports.seriesMulI16Rhs = seriesMulI16Rhs
module.exports.seriesMulI32Rhs = seriesMulI32Rhs
module.exports.seriesMulI64Rhs = seriesMulI64Rhs
module.exports.seriesMulF32Rhs = seriesMulF32Rhs
module.exports.seriesMulF64Rhs = seriesMulF64Rhs
module.exports.seriesRemU8Rhs = seriesRemU8Rhs
module.exports.seriesRemU16Rhs = seriesRemU16Rhs
module.exports.seriesRemU32Rhs = seriesRemU32Rhs
module.exports.seriesRemU64Rhs = seriesRemU64Rhs
module.exports.seriesRemI8Rhs = seriesRemI8Rhs
module.exports.seriesRemI16Rhs = seriesRemI16Rhs
module.exports.seriesRemI32Rhs = seriesRemI32Rhs
module.exports.seriesRemI64Rhs = seriesRemI64Rhs
module.exports.seriesRemF32Rhs = seriesRemF32Rhs
module.exports.seriesRemF64Rhs = seriesRemF64Rhs
module.exports.seriesEqU8 = seriesEqU8
module.exports.seriesEqU16 = seriesEqU16
module.exports.seriesEqU32 = seriesEqU32
module.exports.seriesEqU64 = seriesEqU64
module.exports.seriesEqI8 = seriesEqI8
module.exports.seriesEqI16 = seriesEqI16
module.exports.seriesEqI32 = seriesEqI32
module.exports.seriesEqI64 = seriesEqI64
module.exports.seriesEqF32 = seriesEqF32
module.exports.seriesEqF64 = seriesEqF64
module.exports.seriesEqStr = seriesEqStr
module.exports.seriesNeqU8 = seriesNeqU8
module.exports.seriesNeqU16 = seriesNeqU16
module.exports.seriesNeqU32 = seriesNeqU32
module.exports.seriesNeqU64 = seriesNeqU64
module.exports.seriesNeqI8 = seriesNeqI8
module.exports.seriesNeqI16 = seriesNeqI16
module.exports.seriesNeqI32 = seriesNeqI32
module.exports.seriesNeqI64 = seriesNeqI64
module.exports.seriesNeqF32 = seriesNeqF32
module.exports.seriesNeqF64 = seriesNeqF64
module.exports.seriesNeqStr = seriesNeqStr
module.exports.seriesGtU8 = seriesGtU8
module.exports.seriesGtU16 = seriesGtU16
module.exports.seriesGtU32 = seriesGtU32
module.exports.seriesGtU64 = seriesGtU64
module.exports.seriesGtI8 = seriesGtI8
module.exports.seriesGtI16 = seriesGtI16
module.exports.seriesGtI32 = seriesGtI32
module.exports.seriesGtI64 = seriesGtI64
module.exports.seriesGtF32 = seriesGtF32
module.exports.seriesGtF64 = seriesGtF64
module.exports.seriesGtStr = seriesGtStr
module.exports.seriesGtEqU8 = seriesGtEqU8
module.exports.seriesGtEqU16 = seriesGtEqU16
module.exports.seriesGtEqU32 = seriesGtEqU32
module.exports.seriesGtEqU64 = seriesGtEqU64
module.exports.seriesGtEqI8 = seriesGtEqI8
module.exports.seriesGtEqI16 = seriesGtEqI16
module.exports.seriesGtEqI32 = seriesGtEqI32
module.exports.seriesGtEqI64 = seriesGtEqI64
module.exports.seriesGtEqF32 = seriesGtEqF32
module.exports.seriesGtEqF64 = seriesGtEqF64
module.exports.seriesGtEqStr = seriesGtEqStr
module.exports.seriesLtU8 = seriesLtU8
module.exports.seriesLtU16 = seriesLtU16
module.exports.seriesLtU32 = seriesLtU32
module.exports.seriesLtU64 = seriesLtU64
module.exports.seriesLtI8 = seriesLtI8
module.exports.seriesLtI16 = seriesLtI16
module.exports.seriesLtI32 = seriesLtI32
module.exports.seriesLtI64 = seriesLtI64
module.exports.seriesLtF32 = seriesLtF32
module.exports.seriesLtF64 = seriesLtF64
module.exports.seriesLtStr = seriesLtStr
module.exports.seriesLtEqU8 = seriesLtEqU8
module.exports.seriesLtEqU16 = seriesLtEqU16
module.exports.seriesLtEqU32 = seriesLtEqU32
module.exports.seriesLtEqU64 = seriesLtEqU64
module.exports.seriesLtEqI8 = seriesLtEqI8
module.exports.seriesLtEqI16 = seriesLtEqI16
module.exports.seriesLtEqI32 = seriesLtEqI32
module.exports.seriesLtEqI64 = seriesLtEqI64
module.exports.seriesLtEqF32 = seriesLtEqF32
module.exports.seriesLtEqF64 = seriesLtEqF64
module.exports.seriesLtEqStr = seriesLtEqStr
