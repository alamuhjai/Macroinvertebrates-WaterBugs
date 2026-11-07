# bounds.R: QDA decision regions on two features, for the 4 commonest taxa

pair = c("Area", "Mean.gray.")

top4 = names(sort(table(bugs$Class), decreasing = TRUE))[1:4]
plotData = droplevels(bugs[bugs$Class %in% top4, c("Class", pair)])

savePng("qdaBoundaries.png", width = 1000, height = 800)
partimat(Class ~ Area + Mean.gray., data = plotData, method = "qda",
         main = "QDA boundaries: Area and Mean.gray",
         col.correct = "steelblue", col.wrong = "red")
dev.off()

# error rate on these rows for this feature pair only
fit = qda(Class ~ Area + Mean.gray., data = plotData)
err = mean(predict(fit)$class != plotData$Class)
message("QDA error on ", paste(top4, collapse = ", "), " using ",
        paste(pair, collapse = " + "), ": ", round(err, 3))
writeLines(as.character(round(err, 4)), file.path(outDir, "boundaryError.txt"))
