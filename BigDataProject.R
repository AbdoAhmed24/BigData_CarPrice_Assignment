# ---------------------------
# Install required packages (uncomment if not installed) argoko
# ---------------------------
# install.packages("tidyverse")
# install.packages("caret")
# install.packages("rpart")
# install.packages("rpart.plot")
# install.packages("arules")
# install.packages("cluster")
# install.packages("factoextra")
# install.packages("ggpubr")
# install.packages("glmnet")
# install.packages("car")
# install.packages("randomForest")
# install.packages("corrplot")
# install.packages("e1071")

# car_prices_project_adapted.R
# R 4.5.1 compatible
library(tidyverse)
library(caret)
library(rpart)
library(rpart.plot)
library(arules)
library(cluster)
library(factoextra)
library(ggpubr)
library(glmnet)
library(car) # for VIF
library(randomForest)
library(corrplot)
library(ggplot2)
library(fmsb)
library(e1071)

set.seed(123)

# ---------------------------
# 1. Load data
# ---------------------------
df <- readr::read_csv("CarPrice_Assignment.csv", show_col_types = FALSE)

# Quick look
glimpse(df)
summary(df)
names(df)

# ---------------------------
# 2. Clean & type conversion
# ---------------------------
names(df) <- tolower(names(df))

factor_cols <- c(
  "fueltype", "aspiration", "doornumber", "carbody",
  "drivewheel", "enginelocation", "enginetype",
  "cylindernumber", "fuelsystem"
)
factor_cols <- intersect(factor_cols, names(df))
df[factor_cols] <- map(df[factor_cols], as.factor)

df <- df %>%
  mutate(brand = word(carname, 1)) %>%
  mutate(brand = as.factor(brand))

df <- df %>%
  mutate(
    horsepower = as.numeric(horsepower),
    peakrpm = as.numeric(peakrpm),
    boreratio = as.numeric(boreratio),
    stroke = as.numeric(stroke),
    compressionratio = as.numeric(compressionratio),
    citympg = as.numeric(citympg),
    highwaympg = as.numeric(highwaympg),
    price = as.numeric(price)
  )

cat("NA counts after coercion:\n")
print(colSums(is.na(df)))

# ---------------------------
# 3. Missing values strategy
# ---------------------------
missing_pct <- colSums(is.na(df)) / nrow(df) * 100
print(missing_pct[missing_pct > 0])

num_cols <- names(df)[sapply(df, is.numeric)]
df[num_cols] <- df[num_cols] %>% map_df(~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x))

for (cn in names(df)[sapply(df, is.factor)]) {
  if (any(is.na(df[[cn]]))) {
    mode_val <- names(sort(table(df[[cn]]), decreasing = TRUE))[1]
    df[[cn]][is.na(df[[cn]])] <- mode_val
  }
}

# ---------------------------
# 4. Outliers handling (IQR Method)
# ---------------------------
cap_outliers_iqr <- function(x) {
  if(!is.numeric(x)) return(x)
  q1 <- quantile(x, 0.25, na.rm = TRUE)
  q3 <- quantile(x, 0.75, na.rm = TRUE)
  iqr_val <- q3 - q1
  
  lower_bound <- q1 - 1.5 * iqr_val
  upper_bound <- q3 + 1.5 * iqr_val
  
  x[x < lower_bound] <- lower_bound
  x[x > upper_bound] <- upper_bound
  return(x)
}

# Apply to predictors only (exclude 'price' as high prices are valid)
predictors <- setdiff(num_cols, "price")
df[predictors] <- df[predictors] %>% map_df(cap_outliers_iqr)

cat("IQR Outlier Capping applied to predictors. Price was left untouched.\n")

# ---------------------------
# 5. Feature engineering
# ---------------------------
df <- df %>%
  mutate(price_bucket = cut(price,
    breaks = quantile(price, probs = seq(0, 1, 0.25), na.rm = TRUE),
    include.lowest = TRUE,
    labels = c("Low", "Medium", "High", "VeryHigh")
  )) %>%
  mutate(price_bucket = as.factor(price_bucket))

df <- df %>% mutate(log_price = log(price + 1))

# ---------------------------
# 6. Exploratory Data Analysis (EDA)
# ---------------------------
p1 <- ggplot(df, aes(price)) +
  geom_histogram(bins = 30, alpha = 0.6) +
  ggtitle("Price distribution")
print(p1)

p2 <- ggplot(df, aes(horsepower, price)) +
  geom_point(alpha = 0.6) +
  geom_smooth(method = "loess") +
  ggtitle("Price vs Horsepower")
print(p2)

p3 <- ggplot(df, aes(fueltype, price)) +
  geom_boxplot() +
  ggtitle("Price by Fuel Type")
print(p3)

top_brands <- df %>%
  count(brand, sort = TRUE) %>%
  top_n(10, n)
p4 <- ggplot(df %>% filter(brand %in% top_brands$brand), aes(brand)) +
  geom_bar() +
  ggtitle("Top 10 brands count")
print(p4)

num_df <- df %>% select(where(is.numeric))
cor_mat <- cor(num_df, use = "pairwise.complete.obs")
print(round(cor_mat, 2))
corrplot(cor_mat, method = "color", type = "lower", tl.cex = 0.7)

# ---------------------------
# 7. Hypothesis testing
# ---------------------------
if (all(c("fueltype", "price") %in% names(df))) {
  fuel_counts <- df %>% count(fueltype, sort = TRUE)
  top2fuel <- fuel_counts$fueltype[1:2]
  cat("Top 2 fuels used for t-test:", top2fuel, "\n")
  res_t <- t.test(price ~ fueltype, data = df %>% filter(fueltype %in% top2fuel))
  print(res_t)
}

if ("carbody" %in% names(df)) {
  aov_res <- aov(price ~ carbody, data = df)
  print(summary(aov_res))
  if (summary(aov_res)[[1]]$`Pr(>F)`[1] < 0.05) {
    print(TukeyHSD(aov_res))
  }
}

if (all(c("brand", "drivewheel") %in% names(df))) {
  topb <- df %>%
    count(brand, sort = TRUE) %>%
    slice(1:6) %>%
    pull(brand)
  tab <- table(
    df %>% filter(brand %in% topb) %>% pull(brand),
    df %>% filter(brand %in% topb) %>% pull(drivewheel)
  )
  print(tab)
  print(chisq.test(tab))
}

# ---------------------------
# 8. Prepare dataset for ML
# ---------------------------
ml_vars <- c("log_price", "horsepower", "enginesize", "curbweight", "citympg", "highwaympg", "carwidth", "carlength", "brand")
ml_vars <- intersect(ml_vars, names(df))
ml_df <- df %>%
  select(all_of(ml_vars)) %>%
  na.omit()

# Robust one-hot encoding
# Robust one-hot encoding
dummies <- dummyVars(log_price ~ ., data = ml_df, fullRank = TRUE)
X <- predict(dummies, newdata = ml_df) %>% as.data.frame()
y <- ml_df$log_price
ml_data <- bind_cols(X, log_price = y)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(ml_data$log_price, p = 0.8, list = FALSE)
train <- ml_data[trainIndex, ]
test <- ml_data[-trainIndex, ]

# Ensure test has same columns as train
missing_cols <- setdiff(names(train), names(test))
for (col in missing_cols) test[[col]] <- 0
test <- test[, names(train)]

# ---------------------------
# 9. Models & evaluation
# ---------------------------
# Linear Regression
lm_mod <- lm(log_price ~ ., data = train)
summary(lm_mod)

# Calculate Training Performance (Converted back to Real $)
pred_lm_train_log <- predict(lm_mod, newdata = train)
pred_lm_train_real <- exp(pred_lm_train_log) - 1
# We must compare against the ORIGINAL price, not the log_price
# Since 'train' now has log_price, we approximate the real price with exp(log_price)-1
actual_train_real <- exp(train$log_price) - 1 

lm_train_perf <- postResample(pred_lm_train_real, actual_train_real)
cat("Linear regression TRAINING performance (Real $ Scale):\n"); print(lm_train_perf)

# Calculate Test Performance (Converted back to Real $)
pred_lm_log <- predict(lm_mod, newdata = test)
pred_lm_real <- exp(pred_lm_log) - 1
actual_test_real <- exp(test$log_price) - 1

lm_perf <- postResample(pred_lm_real, actual_test_real)
cat("Linear regression TEST performance (Real $ Scale):\n"); print(lm_perf)



# Decision Tree
tree_mod <- rpart(log_price ~ ., data = train, control = rpart.control(cp = 0.01))
rpart.plot(tree_mod, main = "Decision Tree for Price")
pred_tree_log <- predict(tree_mod, newdata = test)
pred_tree_real <- exp(pred_tree_log) - 1

tree_perf <- postResample(pred_tree_real, actual_test_real)
cat("Decision tree performance (Real $ Scale):\n"); print(tree_perf)

# Random Forest
set.seed(123)
rf_mod <- randomForest(log_price ~ ., data = train, ntree = 200, importance = TRUE)
pred_rf_log <- predict(rf_mod, newdata = test)
pred_rf_real <- exp(pred_rf_log) - 1

rf_perf <- postResample(pred_rf_real, actual_test_real)
cat("Random Forest performance (Real $ Scale):\n"); print(rf_perf)

importance(rf_mod)
varImpPlot(rf_mod)

# Naive Bayes (Classification on Price Bucket)
cat("\n=== Naive Bayes Classification (Predicting Price Range) ===\n")
# Prepare data for classification
nb_vars <- c("price_bucket", "horsepower", "enginesize", "curbweight", "citympg", "highwaympg", "carwidth", "carlength")
nb_df <- df %>% select(all_of(nb_vars)) %>% na.omit()

set.seed(123)
nb_index <- createDataPartition(nb_df$price_bucket, p = 0.8, list = FALSE)
nb_train <- nb_df[nb_index, ]
nb_test  <- nb_df[-nb_index, ]

nb_mod <- naiveBayes(price_bucket ~ ., data = nb_train, usekernel = TRUE)
nb_pred <- predict(nb_mod, nb_test)

nb_cm <- confusionMatrix(nb_pred, nb_test$price_bucket)
print(nb_cm)

# SVM Regression (Tuned)
cat("\n=== SVM Regression (Tuned) ===\n")

# Tune SVM to find best Cost & Gamma
tune_res <- tune(svm, log_price ~ ., data = train, 
                 kernel = "radial", 
                 ranges = list(cost = c(0.1, 1, 10, 100), 
                               gamma = c(0.01, 0.1, 0.5, 1)))

print(tune_res$best.parameters)
svm_best <- tune_res$best.model

pred_svm_log <- predict(svm_best, newdata = test)
pred_svm_real <- exp(pred_svm_log) - 1

svm_perf <- postResample(pred_svm_real, actual_test_real)
cat("Tuned SVM performance (Real $ Scale):\n"); print(svm_perf)

# ---------------------------
# 10. Clustering (K-Means)
# ---------------------------

# Step 1: Prepare clustering data
clus_vars <- c("horsepower", "enginesize", "curbweight", "citympg", "highwaympg", "price")
clus_vars <- intersect(clus_vars, names(df))
clus_df <- na.omit(df[clus_vars])

# Step 2: Standardize the data (K-means is sensitive to scale)
clus_scaled <- scale(clus_df)

# Step 3: Determine Optimal Number of Clusters (Elbow Method)
wss <- numeric(15)
for (k in 1:15) {
  set.seed(123)
  kmeans_result <- kmeans(clus_scaled, centers = k, nstart = 10)
  wss[k] <- kmeans_result$tot.withinss
}

# Plot elbow curve
elbow_data <- data.frame(Clusters = 1:15, WSS = wss)
p_elbow <- ggplot(elbow_data, aes(x = Clusters, y = WSS)) +
  geom_line(color = "blue", linewidth = 1.2) +
  geom_point(color = "red", size = 3) +
  labs(
    title = "Elbow Method for Optimal K",
    x = "Number of Clusters (K)",
    y = "Within-Cluster Sum of Squares (WSS)"
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
print(p_elbow)

# Silhouette method for validation
fviz_nbclust(clus_scaled, kmeans, method = "silhouette")

# Step 4: Apply K-Means Clustering (k=3 based on elbow/silhouette)
k <- 2
set.seed(123)
km <- kmeans(clus_scaled, centers = k, nstart = 25)

# Step 5: Analyze Clusters
cat("Cluster Sizes:\n")
print(table(km$cluster))

cat("\nCluster Centers (Scaled):\n")
print(km$centers)

# Convert scaled centers back to original scale
original_centers <- t(apply(
  km$centers, 1,
  function(r) {
    r * attr(clus_scaled, "scaled:scale") +
      attr(clus_scaled, "scaled:center")
  }
))
cat("\nCluster Centers (Original Scale):\n")
print(original_centers)

# Step 6: Visualize Clusters
# PCA for 2D visualization
pca_result <- prcomp(clus_scaled)
pca_df <- data.frame(
  PC1 = pca_result$x[, 1],
  PC2 = pca_result$x[, 2],
  Cluster = as.factor(km$cluster)
)

p_pca <- ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  stat_ellipse(level = 0.95) +
  labs(
    title = "K-Means Clustering Results (PCA Reduced)",
    x = paste("PC1 (", round(summary(pca_result)$importance[2, 1] * 100, 1), "%)", sep = ""),
    y = paste("PC2 (", round(summary(pca_result)$importance[2, 2] * 100, 1), "%)", sep = "")
  ) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
print(p_pca)

# Factoextra cluster visualization
fviz_cluster(km, data = clus_scaled, main = "K-Means Cluster Visualization")

# Feature pair visualizations
clus_df$cluster <- as.factor(km$cluster)

p_hp_price <- ggplot(clus_df, aes(x = horsepower, y = price, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Clusters: Horsepower vs Price") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
print(p_hp_price)

p_engine_price <- ggplot(clus_df, aes(x = enginesize, y = price, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Clusters: Engine Size vs Price") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))
print(p_engine_price)

# Step 7: Interpret Clusters
cluster_summary <- aggregate(. ~ cluster, data = clus_df, FUN = mean)
cat("\nCluster Summary (Mean Values):\n")
print(cluster_summary)

# Label clusters based on characteristics
cluster_labels <- data.frame(
  cluster = as.factor(1:k),
  label = c("Economy/Mid-Range Cars", "Luxury/Performance Cars")
)
cat("\nCluster Labels:\n")
print(cluster_labels)

# Step 8: Radar Chart for Cluster Characteristics
key_features <- c("horsepower", "enginesize", "curbweight", "citympg", "price")
key_features <- intersect(key_features, names(cluster_summary))
radar_data <- cluster_summary[, c("cluster", key_features)]

# Normalize for radar chart
max_values <- apply(radar_data[, -1], 2, max)
min_values <- apply(radar_data[, -1], 2, min)
radar_scaled <- as.data.frame(scale(radar_data[, -1],
  center = min_values,
  scale = max_values - min_values
))
radar_scaled <- rbind(
  rep(1, ncol(radar_scaled)),
  rep(0, ncol(radar_scaled)),
  radar_scaled
)

# Create radar chart
colors <- c("red", "blue")[1:k]
radarchart(radar_scaled,
  axistype = 1,
  pcol = colors,
  pfcol = adjustcolor(colors, alpha.f = 0.3),
  plwd = 2,
  cglcol = "grey",
  cglty = 1,
  axislabcol = "grey",
  caxislabels = seq(0, 1, 0.25),
  cglwd = 0.8,
  vlcex = 0.8,
  title = "Cluster Characteristics Radar Chart"
)
legend(
  x = 1.2, y = 1,
  legend = paste("Cluster", 1:k, ":", cluster_labels$label),
  bty = "n", pch = 20, col = colors, cex = 0.8
)

# Step 9: ANOVA to validate cluster differences
cat("\n=== ANOVA Results (Checking Cluster Differences) ===\n")
for (feature in c("horsepower", "enginesize", "price")) {
  cat(paste("\nANOVA for", feature, ":\n"))
  anova_test <- aov(clus_df[[feature]] ~ clus_df$cluster)
  print(summary(anova_test))
}

# Step 10: Assign cluster to main dataframe
df$km_cluster <- NA_integer_
df$km_cluster[as.numeric(rownames(clus_df))] <- km$cluster

# Save clustering outputs
write_csv(as.data.frame(original_centers), "cluster_centers.csv")

cat("\n=== CLUSTERING ANALYSIS SUMMARY ===\n")
cat("Number of clusters used:", k, "\n")
cat("Cluster sizes:\n")
print(table(km$cluster))

# ---------------------------
# 11. Association rules
# ---------------------------
trans_df <- df %>%
  mutate(brand_top = if_else(brand %in% (df %>% count(brand) %>% arrange(desc(n)) %>% slice(1:10) %>% pull(brand)),
    as.character(brand), "Other"
  )) %>%
  transmute(price_bucket, brand_top, carbody, fueltype) %>%
  mutate_all(as.factor)

trans <- as(trans_df, "transactions")
rules <- apriori(trans, parameter = list(supp = 0.02, conf = 0.6, minlen = 2))
inspect(sort(rules, by = "lift")[1:20])

# ---------------------------
# 12. Save outputs
# ---------------------------
write_csv(df, "car_prices_cleaned.csv")
write_csv(ml_data, "car_prices_ml_ready.csv")
saveRDS(lm_mod, "lm_model.rds")
saveRDS(tree_mod, "tree_model.rds")
saveRDS(rf_mod, "rf_model.rds")

cat("Script completed. Cleaned data and models saved in working directory.\n")
