# logistic.R: Aselaqua vs Isoperla
# ridge, lasso, elastic net, stepwise logistic and a "relaxed" lasso
# (lasso picks the variables, plain glm refits them).
#
# Everything is scored out-of-sample: the same 10 folds for every model,
# and variable selection (step, lasso) is redone inside each fold.

two = dat %>%
  filter(Class %in% c("Aselaqua", "Isoperla")) %>%
  mutate(Class = factor(Class))

x = model.matrix(Class ~ . - 1, data = two)
y = two$Class
yNum = as.numeric(y == "Isoperla")   # Isoperla is the "event"

set.seed(123)
k = 10
foldId = sample(rep(1:k, length.out = nrow(x)))

modelNames = c("ridge", "lasso", "elastic", "logistic", "relaxed")
alphas = c(ridge = 0, lasso = 1, elastic = 0.5)
pHat = matrix(NA_real_, nrow(x), length(modelNames),
              dimnames = list(NULL, modelNames))

for (f in 1:k) {
  tr = foldId != f
  te = !tr

  # penalised models
  fits = suppressWarnings(lapply(alphas, function(a)
    cv.glmnet(x[tr, ], y[tr], family = "binomial", alpha = a, nfolds = 10)))
  for (m in names(alphas)) {
    pHat[te, m] = as.numeric(predict(fits[[m]], x[te, , drop = FALSE],
                                     s = "lambda.min", type = "response"))
  }

  # stepwise logistic
  full = suppressWarnings(glm(Class ~ ., data = two[tr, ], family = "binomial"))
  stepped = suppressWarnings(step(full, direction = "both", trace = 0))
  pHat[te, "logistic"] = suppressWarnings(
    predict(stepped, newdata = two[te, ], type = "response"))

  # relaxed lasso: refit the variables the lasso kept
  co = coef(fits$lasso, s = "lambda.min")
  keep = setdiff(rownames(co)[as.numeric(co) != 0], "(Intercept)")
  rhs = if (length(keep) == 0) "1" else keep
  relaxed = suppressWarnings(glm(reformulate(rhs, response = "Class"),
                                 data = two[tr, ], family = "binomial"))
  pHat[te, "relaxed"] = suppressWarnings(
    predict(relaxed, newdata = two[te, ], type = "response"))
}

performance = data.frame(
  model = modelNames,
  accuracy = sapply(modelNames, function(m) mean((pHat[, m] > 0.5) == yNum)),
  mse = sapply(modelNames, function(m) mean((yNum - pHat[, m])^2)),
  row.names = NULL
)
performance = performance[order(-performance$accuracy), ]
print(performance)
write.csv(performance, file.path(outDir, "logisticTable.csv"), row.names = FALSE)

# which variables does the lasso keep on the full data?
lassoFull = suppressWarnings(cv.glmnet(x, y, family = "binomial", alpha = 1, nfolds = 10))
co = coef(lassoFull, s = "lambda.min")
chosen = data.frame(variable = rownames(co), coefficient = as.numeric(co))
chosen = chosen[chosen$coefficient != 0 & chosen$variable != "(Intercept)", ]
write.csv(chosen, file.path(outDir, "lassoVariables.csv"), row.names = FALSE)
