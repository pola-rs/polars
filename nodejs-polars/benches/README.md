## Tested on:

- Processor: AMD Ryzen 7 Microsoft Surface (R) Edition 2.00 GHz
- Installed RAM: 16.0 GB
- System type: 64-bit operating system, x64-based processor
- polars version: 0.0.6


## Basic list operation 

Polars series was compared against 
- native array methods
- lodash
- ramda

The tests were run 100 times each to get the results

### Results
|list_size|operation  |data_structure|mean   |min     |max   |stddev  |
|---------|-----------|--------------|-------|--------|------|--------|
|1k       |distinct   |array         |0.0802 |0.058   |0.207 |0.04214 |
|1k       |distinct   |series        |0.118  |0.103   |0.246 |0.0288  |
|1k       |distinct   |lodash        |0.215  |0.107   |1.6   |0.30608 |
|1k       |distinct   |ramda         |0.664  |0.36    |2.01  |0.42527 |
|1k       |intsort    |array         |0.074  |0.0672  |0.199 |0.026229|
|1k       |intsort    |ramda         |0.0761 |0.0685  |0.218 |0.029652|
|1k       |intsort    |series        |0.106  |0.0821  |0.434 |0.07025 |
|1k       |intsort    |lodash        |0.652  |0.249   |2.99  |0.67148 |
|1k       |intsum     |series        |0.00839|0.00348 |0.111 |0.021482|
|1k       |intsum     |array         |0.0319 |0.0228  |0.0703|0.010289|
|1k       |intsum     |lodash        |0.12   |0.0713  |0.487 |0.092209|
|1k       |intsum     |ramda         |0.328  |0.0105  |2.12  |0.47382 |
|1k       |stringsort |series        |0.226  |0.202   |0.523 |0.064543|
|1k       |stringsort |ramda         |0.336  |0.286   |0.608 |0.081377|
|1k       |stringsort |lodash        |0.832  |0.355   |2.82  |0.73076 |
|1k       |stringsort |array         |0.988  |0.297   |17.2  |3.3855  |
|1k       |valueCounts|array         |0.373  |0.264   |0.979 |0.18905 |
|1k       |valueCounts|lodash        |0.442  |0.274   |1.05  |0.2333  |
|1k       |valueCounts|ramda         |1.34   |0.849   |3.24  |0.76985 |
|1k       |valueCounts|series        |7.35   |4.99    |10.5  |1.3679  |
|10k      |distinct   |series        |0.997  |0.702   |2.36  |0.38913 |
|10k      |distinct   |array         |1.04   |0.815   |1.76  |0.25041 |
|10k      |distinct   |lodash        |1.71   |0.995   |2.3   |0.33769 |
|10k      |distinct   |ramda         |5.95   |3.96    |11.3  |1.9389  |
|10k      |intsort    |ramda         |0.573  |0.408   |1.3   |0.21068 |
|10k      |intsort    |array         |0.662  |0.397   |2.11  |0.35309 |
|10k      |intsort    |series        |1.41   |1.12    |1.7   |0.16147 |
|10k      |intsort    |lodash        |1.99   |1.44    |5.52  |0.82568 |
|10k      |intsum     |series        |0.0142 |0.00648 |0.164 |0.031429|
|10k      |intsum     |array         |0.22   |0.159   |0.503 |0.10336 |
|10k      |intsum     |lodash        |0.249  |0.207   |0.526 |0.062575|
|10k      |intsum     |ramda         |3.77   |3.61    |4.17  |0.15662 |
|10k      |stringsort |series        |2.05   |1.76    |2.59  |0.22949 |
|10k      |stringsort |ramda         |2.67   |2.31    |4.17  |0.47361 |
|10k      |stringsort |lodash        |3.69   |3.08    |7.29  |0.9065  |
|10k      |stringsort |array         |4.05   |2.32    |33.2  |6.0867  |
|10k      |valueCounts|series        |0.0054 |0.000511|0.103 |0.020576|
|10k      |valueCounts|lodash        |2.39   |2.2     |4.11  |0.40167 |
|10k      |valueCounts|array         |2.59   |2.39    |3.12  |0.2188  |
|10k      |valueCounts|ramda         |7.18   |6.15    |9.08  |1.0562  |
|100k     |distinct   |series        |8.32   |7.45    |11.6  |1.0923  |
|100k     |distinct   |array         |25.3   |17.1    |35.8  |6.1102  |
|100k     |distinct   |lodash        |27.4   |19.8    |41.3  |5.5639  |
|100k     |distinct   |ramda         |79.2   |65.8    |179.0 |21.492  |
|100k     |intsort    |array         |4.71   |4.17    |8.0   |0.87864 |
|100k     |intsort    |series        |5.0    |4.16    |6.28  |0.54865 |
|100k     |intsort    |ramda         |5.74   |4.56    |10.0  |1.3353  |
|100k     |intsort    |lodash        |20.5   |18.7    |24.9  |1.5702  |
|100k     |intsum     |series        |0.0304 |0.0232  |0.178 |0.030787|
|100k     |intsum     |lodash        |1.57   |1.52    |1.77  |0.074498|
|100k     |intsum     |array         |1.91   |1.58    |3.8   |0.51017 |
|100k     |intsum     |ramda         |41.0   |36.5    |51.8  |4.2231  |
|100k     |stringsort |series        |9.59   |8.65    |10.8  |0.61344 |
|100k     |stringsort |ramda         |33.4   |25.5    |48.9  |6.3622  |
|100k     |stringsort |array         |42.7   |24.8    |384.0 |71.245  |
|100k     |stringsort |lodash        |50.0   |43.8    |60.1  |3.4524  |
|100k     |valueCounts|series        |8.64   |6.41    |10.4  |1.1806  |
|100k     |valueCounts|lodash        |22.4   |21.6    |27.6  |1.2592  |
|100k     |valueCounts|array         |23.3   |22.8    |26.2  |0.7604  |
|100k     |valueCounts|ramda         |55.3   |51.5    |68.6  |4.5847  |
|1M       |distinct   |series        |219.0  |189.0   |453.0 |51.455  |
|1M       |distinct   |array         |526.0  |441.0   |680.0 |85.829  |
|1M       |distinct   |lodash        |563.0  |473.0   |756.0 |86.649  |
|1M       |distinct   |ramda         |1100.0 |927.0   |1450.0|154.78  |
|1M       |intsort    |series        |21.4   |19.5    |23.7  |0.99219 |
|1M       |intsort    |array         |44.3   |41.5    |54.8  |2.6396  |
|1M       |intsort    |ramda         |47.9   |45.0    |54.3  |2.3368  |
|1M       |intsort    |lodash        |457.0  |293.0   |829.0 |180.35  |
|1M       |intsum     |series        |0.573  |0.527   |0.731 |0.060267|
|1M       |intsum     |lodash        |16.1   |15.5    |19.2  |0.90443 |
|1M       |intsum     |array         |16.9   |15.9    |22.0  |1.4031  |
|1M       |intsum     |ramda         |474.0  |380.0   |699.0 |111.62  |
|1M       |stringsort |series        |96.7   |91.6    |110.0 |4.4462  |
|1M       |stringsort |ramda         |397.0  |357.0   |475.0 |31.984  |
|1M       |stringsort |array         |563.0  |348.0   |4770.0|876.43  |
|1M       |stringsort |lodash        |878.0  |651.0   |1390.0|242.73  |
|1M       |valueCounts|series        |19.9   |17.2    |31.2  |2.7456  |
|1M       |valueCounts|lodash        |204.0  |201.0   |218.0 |3.5134  |
|1M       |valueCounts|array         |246.0  |238.0   |275.0 |9.238   |
|1M       |valueCounts|ramda         |529.0  |505.0   |629.0 |30.912  |
___