# explore.R: class counts, Q-Q plots, correlations, PCA

# ---- how many specimens per taxon ----
classCounts = dat %>% count(Class, sort = TRUE)
write.csv(classCounts, file.path(outDir, "classCounts.csv"), row.names = FALSE)

pCounts = ggplot(classCounts, aes(x = reorder(Class, n), y = n, fill = n)) +
  geom_col() +
  geom_text(aes(label = n), hjust = -0.1) +
  coord_flip() +
  expand_limits(y = max(classCounts$n) * 1.1) +
  scale_fill_gradient(low = "lightgreen", high = "darkblue") +
  labs(title = "Specimens per taxon", x = "Taxon", y = "Number of observations") +
  theme_minimal() +
  theme(axis.text = element_text(color = "black"), legend.position = "none")
ggsave(file.path(figDir, "classCounts.png"), pCounts, width = 8, height = 9, dpi = 120)

# ---- Q-Q plots, one per feature ----
savePng("qqFeatures.png", width = 1600, height = 1600, res = 150)
par(mfrow = c(4, 4), bg = "white", mar = c(4, 4, 2, 1))
for (v in featureVars) {
  x = dat[[v]]
  if (is.numeric(x) && length(unique(na.omit(x))) > 1) {
    qqnorm(x, main = v, pch = 20, cex = 0.85, col = "blue")
    qqline(x, col = "red", lwd = 2)
  } else {
    message("Skipping ", v, " (not numeric or constant)")
  }
}
dev.off()

# ---- Mahalanobis distances against a chi-square ----
# if the cloud is roughly elliptical these should sit on the line
X = as.matrix(bugs[, featureVars])
d2 = mahalanobis(X, colMeans(X), cov(X))
savePng("qqMahalanobis.png", width = 900, height = 900)
qqplot(qchisq(ppoints(length(d2)), df = length(featureVars)), d2,
       pch = 20, col = "blue",
       xlab = "Chi-square quantiles", ylab = "Squared Mahalanobis distance",
       main = "Mahalanobis distances vs chi-square")
abline(0, 1, col = "red", lwd = 2)
dev.off()

# ---- correlations ----
corPlot = function(data, file, title, cex) {
  savePng(file, width = 1500, height = 1500, res = 150)
  corrplot(cor(scale(data)), method = "circle", type = "upper",
           order = "hclust", tl.cex = cex, tl.col = "black", diag = FALSE,
           title = title, mar = c(0, 0, 2, 0))
  dev.off()
}

corPlot(bugs[, featureVars], "corSelected.png",
        "Correlation: the 16 features", 0.7)

allNumeric = na.omit(dat[, sapply(dat, is.numeric)])
corPlot(allNumeric, "corAll.png", "Correlation: all numeric columns", 0.6)

# ---- PCA ----
pca = prcomp(bugs[, featureVars], center = TRUE, scale. = TRUE)
explained = pca$sdev^2 / sum(pca$sdev^2)
write.csv(data.frame(pc = seq_along(explained), explained = explained,
                     cumulative = cumsum(explained)),
          file.path(outDir, "pcaVariance.csv"), row.names = FALSE)

savePng("pcaScree.png", width = 800, height = 600)
plot(explained, type = "b", pch = 19,
     xlab = "Principal component", ylab = "Proportion of variance explained",
     main = "Scree plot")
abline(h = 0.05, col = "red", lty = 2)
dev.off()

scores = as.data.frame(pca$x)
scores$Class = bugs$Class
pPca = ggplot(scores, aes(PC1, PC2, color = Class)) +
  geom_point(alpha = 0.6, size = 2) +
  labs(title = "PCA of the image features",
       subtitle = "First two principal components") +
  theme_minimal()
ggsave(file.path(figDir, "pcaScatter.png"), pPca, width = 10, height = 8, dpi = 120)

print(summary(pca))
