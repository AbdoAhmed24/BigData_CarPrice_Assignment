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
df <- readr::read_csv("C:/Uni/Senior 2 Sem1/BigData/Project/CarPrice_Assignment.csv",
                      show_col_types = FALSE)

# Quick look
glimpse(df)
summary(df)
names(df)   # should match the list you provided

# ---------------------------
# 2. Clean & type conversion
# ---------------------------

# Lowercase names (already lowercase in your output but safe)
names(df) <- tolower(names(df))

# Convert some columns to factors
factor_cols <- c("fueltype","aspiration","doornumber","carbody",
                 "drivewheel","enginelocation","enginetype",
                 "cylindernumber","fuelsystem")
factor_cols <- intersect(factor_cols, names(df))
df[factor_cols] <- map(df[factor_cols], as.factor)

# Extract brand from carname (first token)
df <- df %>% mutate(brand = word(carname, 1)) %>% mutate(brand = as.factor(brand))

# Ensure numeric columns are numeric (sometimes read as character)
# Check horsepower and others
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

# Confirm no unexpected NAs introduced
cat("NA counts after coercion:\n")
print(colSums(is.na(df)))

# ---------------------------
# 3. Missing values strategy
# ---------------------------
# This dataset typically has no missing values. But handle if any:
missing_pct <- colSums(is.na(df)) / nrow(df) * 100
print(missing_pct[missing_pct > 0])

# numeric median imputation for any numeric NA
num_cols <- names(df)[sapply(df, is.numeric)]
df[num_cols] <- df[num_cols] %>% map_df(~ ifelse(is.na(.x), median(.x, na.rm = TRUE), .x))

# mode imputation for factor NAs
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
# car age: if there were a 'year' column we'd compute age; not present in this dataset.
# create price_bucket (quartiles) for classification/association rules
df <- df %>%
  mutate(price_bucket = cut(price,
                            breaks = quantile(price, probs = seq(0,1,0.25), na.rm = TRUE),
                            include.lowest = TRUE,
                            labels = c("Low","Medium","High","VeryHigh"))) %>%
  mutate(price_bucket = as.factor(price_bucket))

# optional: log-price for regression if heavily skewed
df <- df %>% mutate(log_price = log(price + 1))

# ---------------------------
# 6. Exploratory Data Analysis (EDA)
# ---------------------------

# 6.1 Price distribution
p1 <- ggplot(df, aes(price)) + geom_histogram(bins = 30, alpha=0.6) + ggtitle("Price distribution")
print(p1)

# 6.2 Price vs horsepower
p2 <- ggplot(df, aes(horsepower, price)) + geom_point(alpha=0.6) + geom_smooth(method="loess") + ggtitle("Price vs Horsepower")
print(p2)

# 6.3 Price by fuel type
p3 <- ggplot(df, aes(fueltype, price)) + geom_boxplot() + ggtitle("Price by Fuel Type")
print(p3)

# 6.4 Top brands frequency
top_brands <- df %>% count(brand, sort = TRUE) %>% top_n(10, n)
p4 <- ggplot(df %>% filter(brand %in% top_brands$brand), aes(brand)) + geom_bar() + ggtitle("Top 10 brands count")
print(p4)

# 6.5 Correlation heatmap for numerics
num_df <- df %>% select(where(is.numeric))
cor_mat <- cor(num_df, use = "pairwise.complete.obs")
print(round(cor_mat, 2))
corrplot(cor_mat, method = "color", type = "lower", tl.cex = 0.7)

# ---------------------------
# 7. Hypothesis testing
# ---------------------------

# H1: Mean price of gas vs diesel differs (two-sample t-test)
if(all(c("fueltype","price") %in% names(df))) {
  # Keep only two largest fuel groups if more than 2
  fuel_counts <- df %>% count(fueltype, sort = TRUE)
  top2fuel <- fuel_counts$fueltype[1:2]
  cat("Top 2 fuels used for t-test:", top2fuel, "\n")
  res_t <- t.test(price ~ fueltype, data = df %>% filter(fueltype %in% top2fuel))
  print(res_t)
}

# H2: ANOVA - price differs by car body
if("carbody" %in% names(df)) {
  aov_res <- aov(price ~ carbody, data = df)
  print(summary(aov_res))
  # If significant, Tukey HSD:
  if(summary(aov_res)[[1]]$`Pr(>F)`[1] < 0.05) {
    print(TukeyHSD(aov_res))
  }
}

# H3: Chi-square between brand (top 6) and drivewheel
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

# Choose relevant predictors (example set)
# We'll use horsepower, enginesize, curbweight, citympg, highwaympg, carwidth, carlength, brand
ml_vars <- c("price","horsepower","enginesize","curbweight","citympg","highwaympg","carwidth","carlength","brand")
ml_vars <- intersect(ml_vars, names(df))
ml_df <- df %>% select(all_of(ml_vars)) %>% na.omit()

# One-hot encode factors via caret's dummyVars
dummies <- dummyVars(price ~ ., data = ml_df)
X <- predict(dummies, newdata = ml_df) %>% as.data.frame()
y <- ml_df$price
ml_data <- bind_cols(X, price = y)

# Train-test split
trainIndex <- createDataPartition(ml_data$price, p = .8, list = FALSE)
train <- ml_data[trainIndex,]
test  <- ml_data[-trainIndex,]

# ---------------------------
# 9. Models & evaluation
# ---------------------------

# 9.1 Linear regression
lm_mod <- lm(price ~ ., data = train)
summary(lm_mod)
pred_lm <- predict(lm_mod, newdata = test)
lm_perf <- postResample(pred_lm, test$price)
cat("Linear regression performance (RMSE, R-squared):\n"); print(lm_perf)

# Check VIF for multicollinearity
vif_vals <- vif(lm_mod)
print(vif_vals)

# Residual plots
plot(lm_mod, which = 1)  # Residuals vs fitted
plot(lm_mod, which = 2)  # QQ plot

# 9.2 LASSO (glmnet) via caret
train_ctrl <- trainControl(method = "cv", number = 5)
set.seed(42)
lasso_mod <- train(
  price ~ .,
  data = train,
  method = "glmnet",
  trControl = train_ctrl,
  tuneLength = 10
)
print(lasso_mod)
pred_lasso <- predict(lasso_mod, newdata = test)
lasso_perf <- postResample(pred_lasso, test$price)
cat("LASSO performance:\n"); print(lasso_perf)

# 9.3 Decision tree
tree_mod <- rpart(price ~ ., data = train, control = rpart.control(cp = 0.01))
rpart.plot(tree_mod, main = "Decision Tree for Price")
pred_tree <- predict(tree_mod, newdata = test)
tree_perf <- postResample(pred_tree, test$price)
cat("Decision tree performance:\n"); print(tree_perf)

# 9.4 Random forest (optional but often strong)
set.seed(123)
rf_mod <- randomForest(price ~ ., data = train, ntree = 200, importance = TRUE)
pred_rf <- predict(rf_mod, newdata = test)
rf_perf <- postResample(pred_rf, test$price)
cat("Random Forest performance:\n"); print(rf_perf)
importance(rf_mod)
varImpPlot(rf_mod)

# ---------------------------
# 10. Clustering (K-means)
# ---------------------------
clus_vars <- c("horsepower","enginesize","curbweight","price")
clus_vars <- intersect(clus_vars, names(df))
clus_df <- na.omit(df[clus_vars])
clus_scaled <- scale(clus_df)

# Determine k (silhouette)
fviz_nbclust(clus_scaled, kmeans, method = "silhouette")
# pick k = 3 (example)
km <- kmeans(clus_scaled, centers = 3, nstart = 25)
table(km$cluster)
fviz_cluster(km, data = clus_scaled)

# Add cluster labels to original df
df$km_cluster <- NA
df$km_cluster[rownames(clus_df)] <- km$cluster

# ---------------------------
# 11. Association rules (categorical)
# ---------------------------
# Use discrete categories: price_bucket, brand (top), carbody, fueltype
trans_df <- df %>%
  mutate(brand_top = if_else(brand %in% (df %>% count(brand) %>% arrange(desc(n)) %>% slice(1:10) %>% pull(brand)),
                             as.character(brand), "Other")) %>%
  transmute(price_bucket, brand_top, carbody, fueltype) %>%
  mutate_all(as.factor)

# Convert to transactions and mine rules
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
