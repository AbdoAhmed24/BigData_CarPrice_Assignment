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
library(car)         # for VIF
library(randomForest)
library(corrplot)

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

factor_cols <- c("fueltype","aspiration","doornumber","carbody",
                 "drivewheel","enginelocation","enginetype",
                 "cylindernumber","fuelsystem")
factor_cols <- intersect(factor_cols, names(df))
df[factor_cols] <- map(df[factor_cols], as.factor)

df <- df %>% mutate(brand = word(carname, 1)) %>% mutate(brand = as.factor(brand))

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

for(cn in names(df)[sapply(df, is.factor)]) {
  if (any(is.na(df[[cn]]))) {
    mode_val <- names(sort(table(df[[cn]]), decreasing = TRUE))[1]
    df[[cn]][is.na(df[[cn]])] <- mode_val
  }
}

# ---------------------------
# 4. Outliers handling (cap at 1%/99%)
# ---------------------------
cap_at_quantiles <- function(x) {
  if(!is.numeric(x)) return(x)
  q <- quantile(x, probs = c(0.01, 0.99), na.rm = TRUE)
  x[x < q[1]] <- q[1]
  x[x > q[2]] <- q[2]
  return(x)
}
df[num_cols] <- df[num_cols] %>% map_df(cap_at_quantiles)

# ---------------------------
# 5. Feature engineering
# ---------------------------
df <- df %>%
  mutate(price_bucket = cut(price,
                            breaks = quantile(price, probs = seq(0,1,0.25), na.rm = TRUE),
                            include.lowest = TRUE,
                            labels = c("Low","Medium","High","VeryHigh"))) %>%
  mutate(price_bucket = as.factor(price_bucket))

df <- df %>% mutate(log_price = log(price + 1))

# ---------------------------
# 6. Exploratory Data Analysis (EDA)
# ---------------------------
p1 <- ggplot(df, aes(price)) + geom_histogram(bins = 30, alpha=0.6) + ggtitle("Price distribution")
print(p1)

p2 <- ggplot(df, aes(horsepower, price)) + geom_point(alpha=0.6) + geom_smooth(method="loess") + ggtitle("Price vs Horsepower")
print(p2)

p3 <- ggplot(df, aes(fueltype, price)) + geom_boxplot() + ggtitle("Price by Fuel Type")
print(p3)

top_brands <- df %>% count(brand, sort = TRUE) %>% top_n(10, n)
p4 <- ggplot(df %>% filter(brand %in% top_brands$brand), aes(brand)) + geom_bar() + ggtitle("Top 10 brands count")
print(p4)

num_df <- df %>% select(where(is.numeric))
cor_mat <- cor(num_df, use = "pairwise.complete.obs")
print(round(cor_mat, 2))
corrplot(cor_mat, method = "color", type = "lower", tl.cex = 0.7)

# ---------------------------
# 7. Hypothesis testing
# ---------------------------
if(all(c("fueltype","price") %in% names(df))) {
  fuel_counts <- df %>% count(fueltype, sort = TRUE)
  top2fuel <- fuel_counts$fueltype[1:2]
  cat("Top 2 fuels used for t-test:", top2fuel, "\n")
  res_t <- t.test(price ~ fueltype, data = df %>% filter(fueltype %in% top2fuel))
  print(res_t)
}

if("carbody" %in% names(df)) {
  aov_res <- aov(price ~ carbody, data = df)
  print(summary(aov_res))
  if(summary(aov_res)[[1]]$`Pr(>F)`[1] < 0.05) {
    print(TukeyHSD(aov_res))
  }
}

if(all(c("brand","drivewheel") %in% names(df))) {
  topb <- df %>% count(brand, sort = TRUE) %>% slice(1:6) %>% pull(brand)
  tab <- table(df %>% filter(brand %in% topb) %>% pull(brand),
               df %>% filter(brand %in% topb) %>% pull(drivewheel))
  print(tab)
  print(chisq.test(tab))
}

# ---------------------------
# 8. Prepare dataset for ML
# ---------------------------
ml_vars <- c("price","horsepower","enginesize","curbweight","citympg","highwaympg","carwidth","carlength","brand")
ml_vars <- intersect(ml_vars, names(df))
ml_df <- df %>% select(all_of(ml_vars)) %>% na.omit()

# Robust one-hot encoding
dummies <- dummyVars(price ~ ., data = ml_df, fullRank = TRUE)
X <- predict(dummies, newdata = ml_df) %>% as.data.frame()
y <- ml_df$price
ml_data <- bind_cols(X, price = y)

# Train-test split
set.seed(123)
trainIndex <- createDataPartition(ml_data$price, p = 0.8, list = FALSE)
train <- ml_data[trainIndex, ]
test  <- ml_data[-trainIndex, ]

# Ensure test has same columns as train
missing_cols <- setdiff(names(train), names(test))
for(col in missing_cols) test[[col]] <- 0
test <- test[, names(train)]

# ---------------------------
# 9. Models & evaluation
# ---------------------------
# Linear Regression
lm_mod <- lm(price ~ ., data = train)
summary(lm_mod)
pred_lm <- predict(lm_mod, newdata = test)
lm_perf <- postResample(pred_lm, test$price)
cat("Linear regression performance (RMSE, R-squared, MAE):\n"); print(lm_perf)

# VIF (robust to aliased coefficients)
coefs_na <- names(coef(lm_mod))[is.na(coef(lm_mod))]
if(length(coefs_na) > 0){
  cat("Aliased coefficients detected and removed for VIF calculation:\n")
  print(coefs_na)
  train_vif <- train %>% select(-all_of(coefs_na))
  lm_mod_vif <- lm(price ~ ., data = train_vif)
  vif_vals <- vif(lm_mod_vif)
} else {
  vif_vals <- vif(lm_mod)
}
print(vif_vals)

plot(lm_mod, which = 1)
plot(lm_mod, which = 2)

# LASSO
train_ctrl <- trainControl(method = "cv", number = 5)
set.seed(42)
lasso_mod <- train(price ~ ., data = train, method = "glmnet", trControl = train_ctrl, tuneLength = 10)
pred_lasso <- predict(lasso_mod, newdata = test)
lasso_perf <- postResample(pred_lasso, test$price)
cat("LASSO performance:\n"); print(lasso_perf)

# Decision Tree
tree_mod <- rpart(price ~ ., data = train, control = rpart.control(cp = 0.01))
rpart.plot(tree_mod, main = "Decision Tree for Price")
pred_tree <- predict(tree_mod, newdata = test)
tree_perf <- postResample(pred_tree, test$price)
cat("Decision tree performance:\n"); print(tree_perf)

# Random Forest
set.seed(123)
rf_mod <- randomForest(price ~ ., data = train, ntree = 200, importance = TRUE)
pred_rf <- predict(rf_mod, newdata = test)
rf_perf <- postResample(pred_rf, test$price)
cat("Random Forest performance:\n"); print(rf_perf)
importance(rf_mod)
varImpPlot(rf_mod)

# ---------------------------
# 10. Clustering
# ---------------------------
clus_vars <- c("horsepower","enginesize","curbweight","price")
clus_vars <- intersect(clus_vars, names(df))
clus_df <- na.omit(df[clus_vars])
clus_scaled <- scale(clus_df)

fviz_nbclust(clus_scaled, kmeans, method = "silhouette")
km <- kmeans(clus_scaled, centers = 3, nstart = 25)
table(km$cluster)
fviz_cluster(km, data = clus_scaled)

# Initialize cluster column
df$km_cluster <- NA_integer_

# Assign only for the rows actually used in clustering
df$km_cluster[as.numeric(rownames(clus_df))] <- km$cluster

# ---------------------------
# 11. Association rules
# ---------------------------
trans_df <- df %>%
  mutate(brand_top = if_else(brand %in% (df %>% count(brand) %>% arrange(desc(n)) %>% slice(1:10) %>% pull(brand)),
                             as.character(brand), "Other")) %>%
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
saveRDS(lasso_mod, "lasso_model.rds")
saveRDS(tree_mod, "tree_model.rds")
saveRDS(rf_mod, "rf_model.rds")

cat("Script completed. Cleaned data and models saved in working directory.\n")
