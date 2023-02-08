# Rscript groupby-datagen.R 1e7 1e2 0 0 ## 1e7 rows, 1e2 K, 0% NAs, random order
# Rscript groupby-datagen.R 1e8 1e1 5 1 ## 1e8 rows, 10 K, 5% NAs, sorted order
args = commandArgs(TRUE)

pretty_sci = function(x) {
  tmp<-strsplit(as.character(x), "+", fixed=TRUE)[[1L]]
  if(length(tmp)==1L) {
    paste0(substr(tmp, 1L, 1L), "e", nchar(tmp)-1L)
  } else if(length(tmp)==2L){
    paste0(tmp[1L], as.character(as.integer(tmp[2L])))
  }
}

library(data.table)
N=as.integer(args[1L]); K=as.integer(args[2L]); nas=as.integer(args[3L]); sort=as.integer(args[4L])
stopifnot(nas<=100L, nas>=0L, sort%in%c(0L,1L))
set.seed(108)
cat(sprintf("Producing data of %s rows, %s K groups factors, %s NAs ratio, %s sort flag\n", pretty_sci(N), pretty_sci(K), nas, sort))
DT = list()
DT[["id1"]] = sample(sprintf("id%03d",1:K), N, TRUE)      # large groups (char)
DT[["id2"]] = sample(sprintf("id%03d",1:K), N, TRUE)      # small groups (char)
DT[["id3"]] = sample(sprintf("id%010d",1:(N/K)), N, TRUE) # large groups (char)
DT[["id4"]] = sample(K, N, TRUE)                          # large groups (int)
DT[["id5"]] = sample(K, N, TRUE)                          # small groups (int)
DT[["id6"]] = sample(N/K, N, TRUE)                        # small groups (int)
DT[["v1"]] =  sample(5, N, TRUE)                          # int in range [1,5]
DT[["v2"]] =  sample(15, N, TRUE)                         # int in range [1,15]
DT[["v3"]] =  round(runif(N,max=100),6)                   # numeric e.g. 23.574912
setDT(DT)
if (nas>0L) {
  cat("Inputting NAs\n")
  for (col in paste0("id",1:6)) {
    ucol = unique(DT[[col]])
    nna = as.integer(length(ucol) * (nas/100))
    if (nna)
      set(DT, DT[.(sample(ucol, nna)), on=col, which=TRUE], col, NA)
    rm(ucol)
  }
  nna = as.integer(nrow(DT) * (nas/100))
  if (nna) {
    for (col in paste0("v",1:3))
      set(DT, sample(nrow(DT), nna), col, NA)
  }
}
if (sort==1L) {
  cat("Sorting data\n")
  setkeyv(DT, paste0("id", 1:6))
}
file = sprintf("G1_%s_%s_%s_%s.csv", pretty_sci(N), pretty_sci(K), nas, sort)
cat(sprintf("Writing data to %s\n", file))
fwrite(DT, file)
cat(sprintf("Data written to %s, quitting\n", file))
if (!interactive()) quit("no", status=0)
