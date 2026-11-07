# models.R: compare classifiers with repeated CV, then test the winner
#
# Order of things matters here: the 80/20 split happens first, and the
# feature filtering is learned on the training part only, so the held-out
# 20% never influences which columns are kept.

allData = na.omit(dat)

# ---- split ----
trainIdx  = createDataPartition(allData$Class, p = 0.8, list = FALSE)
trainData = allData[trainIdx, ]
testData  = allData[-trainIdx, ]

dropCols = function(cols) {
  trainData <<- trainData[, !(names(trainData) %in% cols)]
  testData  <<- testData[,  !(names(testData)  %in% cols)]
}

# ---- filter features (training data only) ----
nzv = nearZeroVar(trainData, names = TRUE)
if (length(nzv) > 0) dropCols(nzv)

numCols = names(trainData)[sapply(trainData, is.numeric)]
highCor = findCorrelation(cor(trainData[, numCols], use = "pairwise.complete.obs"),
                          cutoff = 0.75, names = TRUE)
if (length(highCor) > 0) dropCols(highCor)

numCols = names(trainData)[sapply(trainData, is.numeric)]
combo = findLinearCombos(trainData[, numCols])
if (!is.null(combo$remove)) dropCols(numCols[combo$remove])

keptFeatures = setdiff(names(trainData), "Class")
writeLines(keptFeatures, file.path(outDir, "keptFeatures.txt"))
message("Features kept: ", paste(keptFeatures, collapse = ", "))

# ---- fit everything the same way ----
cv = trainControl(method = "repeatedcv", number = 10, repeats = 5,
                  savePredictions = "final")

fitModel = function(method, pre = NULL, ...) {
  suppressWarnings(
    train(Class ~ ., data = trainData, method = method, trControl = cv,
          metric = "Accuracy", preProcess = pre, ...)
  )
}

models = list(
  QDA      = fitModel("qda",   c("center", "scale")),
  RF       = fitModel("rf",    tuneLength = 5, importance = TRUE),
  RF_PCA   = fitModel("rf",    c("center", "scale", "pca"), tuneLength = 5, importance = TRUE),
  kNN      = fitModel("knn",   c("center", "scale"), tuneLength = 15),
  kNN_PCA  = fitModel("knn",   c("center", "scale", "pca"), tuneLength = 15),
  LDA      = fitModel("lda"),
  Tree     = fitModel("rpart", tuneLength = 15)
)

# ---- cross-validated comparison ----
perf = do.call(rbind, lapply(names(models), function(nm) {
  p = getTrainPerf(models[[nm]])
  data.frame(model = nm, accuracy = p$TrainAccuracy, kappa = p$TrainKappa,
             error = 1 - p$TrainAccuracy)
}))
perf = perf[order(-perf$accuracy), ]
print(perf)
write.csv(perf, file.path(outDir, "cvTable.csv"), row.names = FALSE)

savePng("cvBoxplot.png", width = 1000, height = 700)
print(bwplot(resamples(models)))
dev.off()

bestName = perf$model[1]
best = models[[bestName]]
message("Best model by CV accuracy: ", bestName)

# ---- 20% hold-out ----
pred = predict(best, newdata = testData)
cm = confusionMatrix(pred, testData$Class)
write.csv(as.matrix(cm$table), file.path(outDir, "confusionMatrix.csv"))
writeLines(capture.output(print(cm)), file.path(outDir, "confusionMatrix.txt"))

macroP  = mean(cm$byClass[, "Pos Pred Value"], na.rm = TRUE)
macroR  = mean(cm$byClass[, "Sensitivity"],    na.rm = TRUE)
holdout = data.frame(
  model = bestName,
  accuracy = unname(cm$overall["Accuracy"]),
  kappa = unname(cm$overall["Kappa"]),
  macroPrecision = macroP,
  macroRecall = macroR,
  macroF1 = 2 * macroP * macroR / (macroP + macroR),
  balancedAccuracy = mean(cm$byClass[, "Balanced Accuracy"], na.rm = TRUE)
)
write.csv(holdout, file.path(outDir, "holdout.csv"), row.names = FALSE)

holdoutSummary = data.frame(
  model = bestName,
  accuracy = round(unname(cm$overall["Accuracy"]), 4),
  ci95 = sprintf("[%.4f, %.4f]", cm$overall["AccuracyLower"], cm$overall["AccuracyUpper"]),
  kappa = round(unname(cm$overall["Kappa"]), 4),
  macroPrecision = round(macroP, 4),
  macroRecall = round(macroR, 4),
  macroF1 = round(2 * macroP * macroR / (macroP + macroR), 4),
  balancedAccuracy = round(mean(cm$byClass[, "Balanced Accuracy"], na.rm = TRUE), 4)
)
message("\nHold-out summary (20% test set):")
print(holdoutSummary, row.names = FALSE)

byTaxon = data.frame(
  taxon = sub("^Class: ", "", rownames(cm$byClass)),
  sensitivity = round(cm$byClass[, "Sensitivity"], 4),
  precision = round(cm$byClass[, "Pos Pred Value"], 4),
  balancedAccuracy = round(cm$byClass[, "Balanced Accuracy"], 4),
  row.names = NULL
)
byTaxon = byTaxon[order(-byTaxon$sensitivity, -byTaxon$balancedAccuracy), ]
message("\nBest-classified taxa (top 5):")
print(head(byTaxon, 5), row.names = FALSE)
message("\nWorst-classified taxa (bottom 5):")
print(tail(byTaxon, 5), row.names = FALSE)

# ---- how sure is QDA? (on the training rows, so this is optimistic) ----
qdaProbs = predict(models$QDA, newdata = trainData, type = "prob")
top2 = t(apply(qdaProbs, 1, function(p) sort(p, decreasing = TRUE)[1:2]))
conf = data.frame(taxon = trainData$Class, maxProb = top2[, 1],
                  margin = top2[, 1] - top2[, 2])
pConf = ggplot(conf, aes(maxProb, margin, color = taxon)) +
  geom_point(alpha = 0.5) +
  labs(title = "QDA confidence (training rows)", x = "Top probability",
       y = "Margin over runner-up") +
  theme_minimal()
ggsave(file.path(figDir, "qdaConfidence.png"), pConf, width = 9, height = 6, dpi = 120)

# ---- predictions for the unlabelled test file ----
if (file.exists(testFile)) {
  newBugs = read.table(testFile, header = TRUE)
  newBugs = newBugs[, keptFeatures, drop = FALSE]
  out = data.frame(predictedClass = predict(best, newdata = newBugs))
  write.csv(out, file.path(outDir, "predictions.csv"), row.names = FALSE)
} else {
  message("No ", testFile, ", skipping predictions.")
}
