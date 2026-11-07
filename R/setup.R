# setup.R: packages, seed, paths and the data. Every other script starts here.

suppressPackageStartupMessages({
  library(ggplot2)
  library(dplyr)
  library(corrplot)
  library(randomForest)
  library(caret)
  library(MASS)
  library(klaR)
  library(glmnet)
})

set.seed(734985)

# paths (run everything from the repo root)
trainFile = "data/waterbugs_training.dat.txt"
testFile  = "data/waterbugs_testing.dat.txt"
figDir    = "figures"
outDir    = "results"

for (d in c(figDir, outDir)) dir.create(d, showWarnings = FALSE)

if (!file.exists(trainFile)) {
  stop("Can't find ", trainFile, ". Put the training file in data/ (see README).")
}

# the 16 numeric image features, defined once and used everywhere
featureVars = c(
  "Area", "Perim", "Width", "Height", "Major", "Minor",
  "Angle", "Circ", "Feret",
  "Mean.gray.", "StdDev.gray.", "Mode.gray.",
  "IntDen.gray.", "Median.gray.", "Skew.gray.", "Kurt.gray."
)

dat = read.table(trainFile, header = TRUE)
dat$Class = as.factor(dat$Class)

# complete rows only, features plus class
bugs = na.omit(dat[, c(featureVars, "Class")])

# small helper so every figure is saved the same way
savePng = function(name, width = 1000, height = 800, res = 120) {
  png(file.path(figDir, name), width = width, height = height, res = res)
}
