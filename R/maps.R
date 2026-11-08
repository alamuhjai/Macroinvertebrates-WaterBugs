# maps.R: class maps (PAC against farness)
#
# PAC  = how close the runner-up class is to the true class (high = ambiguous)
# farness = how far a specimen sits from the centre of its own class
#
# PAC comes from out-of-bag random forest probabilities, so each specimen
# is scored by trees that never saw it.

classMap = function(data, title, file, outlierCut = 0.95) {
  x = scale(data[, featureVars])
  y = data$Class
  n = nrow(x)

  rf = randomForest(Class ~ ., data = data, ntree = 500)
  post = predict(rf, type = "prob")   # no newdata means out-of-bag

  pac = numeric(n)
  for (i in 1:n) {
    trueP = post[i, as.character(y[i])]
    altP  = max(post[i, colnames(post) != as.character(y[i])])
    pac[i] = altP / (trueP + altP + 1e-12)
  }

  farness = rep(NA_real_, n)
  for (i in 1:n) {
    xi = x[y == y[i], , drop = FALSE]
    if (nrow(xi) < 3) next   # too few specimens to estimate a covariance
    covMat = cov(xi) + diag(1e-6, ncol(x))
    farness[i] = mahalanobis(x[i, ], colMeans(xi), covMat)
  }
  farness = rank(farness, na.last = "keep") / sum(!is.na(farness))
  outlier = !is.na(farness) & farness > outlierCut

  savePng(file, width = 1400, height = 900, res = 130)
  par(bg = "white", xpd = TRUE, mar = c(5, 4, 4, 11))
  plot(farness, pac, col = as.numeric(y), pch = ifelse(outlier, 17, 19),
       xlab = "Farness", ylab = "PAC", main = title)
  abline(h = 0.2, lty = 2)
  abline(v = 0.8, lty = 2)
  legend("right", legend = levels(y), col = seq_along(levels(y)), pch = 19,
         cex = 0.68, bty = "n", title = "Taxon", inset = c(-0.22, 0))
  legend("topleft", legend = c("Normal point", "Outlier"), pch = c(19, 17),
         bty = "n")
  dev.off()

  invisible(data.frame(pac = pac, farness = farness, outlier = outlier))
}

counts = sort(table(bugs$Class), decreasing = TRUE)
top8   = names(counts)[1:8]
last8  = names(tail(counts, 8))

classMap(bugs, "Class map: all taxa", "classMapAll.png")
classMap(droplevels(bugs[bugs$Class %in% top8, ]),
         "Class map: 8 most common taxa", "classMapTop8.png")
classMap(droplevels(bugs[bugs$Class %in% last8, ]),
         "Class map: 8 rarest taxa", "classMapLast8.png")
