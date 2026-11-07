# run.R: runs the whole analysis, in order. Start R in the repo root.
#
#   Rscript run.R
#
# models.R is the slow one (repeated CV on seven models).

source("R/setup.R")

steps = c("explore", "maps", "models", "bounds", "logistic")
for (s in steps) {
  message("\n== ", s, " ==")
  suppressWarnings(source(file.path("R", paste0(s, ".R")), echo = FALSE))
}

message("\nDone. Figures are in ", figDir, "/ and tables in ", outDir, "/.")
