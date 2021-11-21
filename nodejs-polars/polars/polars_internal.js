/* eslint-disable camelcase */
const {loadBinding} = require('@node-rs/helper')
const path = require('path')
const up1 = path.join(__dirname, '../')
const polars_internal = loadBinding(up1, 'nodejs_polars', 'nodejs_polars')
module.exports = polars_internal
