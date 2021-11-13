import Dataframe from './dataframe';
import Series from './series';
import {PolarsDataType} from './datatypes';
import * as io from './io';
import * as convert from './convert';

export default {
  Dataframe: Dataframe.from,
  Series: Series.from,
  ...io,
  ...convert,
  ...PolarsDataType
}