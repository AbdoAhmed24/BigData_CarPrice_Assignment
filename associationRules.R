# ==============================================
# PHASE 4: ASSOCIATION RULES (APRIORI ALGORITHM)
# ==============================================

#install.packages("arules")
#install.packages("arulesViz")

# Load necessary libraries
library(arules)
library(arulesViz)
library(ggplot2)
library(dplyr)

# -------------------------------------------------
# Step 1: Prepare Categorical Data for Association Rules
# -------------------------------------------------

# Load the clustered data
car_data <- read.csv("C:/hassan/cess/semester9/big data/project/BigData_CarPrice_Assignment-main/BigData_CarPrice_Assignment-main/car_data_with_clusters.csv")
# Create categorical variables from continuous ones (binning)
# This is necessary because Apriori works best with categorical data

# Bin horsepower
car_data$horsepower_cat <- cut(car_data$horsepower,
                               breaks = c(0, 100, 150, 200, 250),
                               labels = c("Low_HP", "Medium_HP", "High_HP", "VeryHigh_HP"),
                               include.lowest = TRUE)

# Bin engine size
car_data$enginesize_cat <- cut(car_data$enginesize,
                               breaks = c(0, 100, 150, 200, 300, 400),
                               labels = c("Small_Engine", "Medium_Engine", "Large_Engine", "VLarge_Engine", "XLarge_Engine"),
                               include.lowest = TRUE)

# Bin price
car_data$price_cat <- cut(car_data$price,
                          breaks = c(0, 10000, 20000, 30000, 50000),
                          labels = c("Budget", "MidRange", "Premium", "Luxury"),
                          include.lowest = TRUE)

# Bin fuel efficiency
car_data$mpg_cat <- cut(car_data$citympg,
                        breaks = c(0, 20, 25, 30, 50),
                        labels = c("Low_MPG", "Medium_MPG", "High_MPG", "VeryHigh_MPG"),
                        include.lowest = TRUE)

# Bin curb weight
car_data$weight_cat <- cut(car_data$curbweight,
                           breaks = c(0, 2000, 2500, 3000, 4000),
                           labels = c("Light", "Medium", "Heavy", "VeryHeavy"),
                           include.lowest = TRUE)

# Create brand category (single brand per car)
car_data$brand <- NA

# Find which brand column has value 1 for each row
brand_cols <- grep("^brand\\.", names(car_data), value = TRUE)

for (i in 1:nrow(car_data)) {
  brand_index <- which(car_data[i, brand_cols] == 1)
  if (length(brand_index) > 0) {
    car_data$brand[i] <- gsub("brand\\.", "", brand_cols[brand_index[1]])
  } else {
    car_data$brand[i] <- "Unknown"
  }
}

# Convert brand to factor
car_data$brand <- as.factor(car_data$brand)

# -------------------------------------------------
# Step 2: Create Transaction Data
# -------------------------------------------------

# Select categorical columns for association rules
cat_cols <- c("horsepower_cat", "enginesize_cat", "price_cat", 
              "mpg_cat", "weight_cat", "brand", "label")

# Create transaction data
trans_data <- car_data[, cat_cols]

# Convert all columns to factors
trans_data <- as.data.frame(lapply(trans_data, as.factor))

# Convert to transactions format
transactions <- as(trans_data, "transactions")

# Summary of transactions
cat("\n=== TRANSACTION SUMMARY ===\n")
summary(transactions)

# Visualize item frequency
itemFrequencyPlot(transactions, 
                  topN = 15,
                  type = "relative",
                  main = "Top 15 Most Frequent Items",
                  col = rainbow(15),
                  ylab = "Item Frequency (Relative)")

# -------------------------------------------------
# Step 3: Apply Apriori Algorithm
# -------------------------------------------------

# Mine association rules
rules <- apriori(transactions,
                 parameter = list(
                   support = 0.1,     # Minimum support (10% of transactions)
                   confidence = 0.7,  # Minimum confidence (70%)
                   minlen = 2,        # Minimum rule length
                   maxlen = 4         # Maximum rule length
                 ))

cat("\n=== ASSOCIATION RULES SUMMARY ===\n")
cat("Number of rules found:", length(rules), "\n")

# -------------------------------------------------
# Step 4: Filter and Sort Rules
# -------------------------------------------------

# Remove redundant rules
rules <- rules[!is.redundant(rules)]

# Sort rules by different metrics
rules_by_lift <- sort(rules, by = "lift", decreasing = TRUE)
rules_by_confidence <- sort(rules, by = "confidence", decreasing = TRUE)
rules_by_support <- sort(rules, by = "support", decreasing = TRUE)

# -------------------------------------------------
# Step 5: Inspect and Interpret Rules
# -------------------------------------------------

cat("\n=== TOP 10 RULES BY LIFT ===\n")
inspect(head(rules_by_lift, 10))

cat("\n=== TOP 10 RULES BY CONFIDENCE ===\n")
inspect(head(rules_by_confidence, 10))

cat("\n=== TOP 10 RULES BY SUPPORT ===\n")
inspect(head(rules_by_support, 10))

# -------------------------------------------------
# Step 6: Generate Specific Rule Types
# -------------------------------------------------

# Rules that lead to specific car segments
luxury_rules <- subset(rules, rhs %in% c("label=Luxury Cars", "price_cat=Luxury", "price_cat=Premium"))
economy_rules <- subset(rules, rhs %in% c("label=Economy Cars", "price_cat=Budget"))
performance_rules <- subset(rules, rhs %in% c("label=Performance Cars"))

cat("\n=== RULES RELATED TO LUXURY CARS ===\n")
if (length(luxury_rules) > 0) {
  inspect(head(sort(luxury_rules, by = "lift"), 10))
} else {
  cat("No specific luxury rules found. Try lowering support/confidence thresholds.\n")
}

cat("\n=== RULES RELATED TO ECONOMY CARS ===\n")
if (length(economy_rules) > 0) {
  inspect(head(sort(economy_rules, by = "lift"), 10))
}

cat("\n=== RULES RELATED TO PERFORMANCE CARS ===\n")
if (length(performance_rules) > 0) {
  inspect(head(sort(performance_rules, by = "lift"), 10))
}

# -------------------------------------------------
# Step 7: Visualize Association Rules
# -------------------------------------------------

# Scatter plot of rules
plot(rules, 
     method = "scatterplot",
     measure = c("support", "confidence"),
     shading = "lift",
     main = "Association Rules: Support vs Confidence")

# Grouped matrix plot
plot(head(rules_by_lift, 20),
     method = "grouped",
     control = list(k = 5),
     main = "Top 20 Rules by Lift (Grouped Matrix)")

# Graph visualization
plot(head(rules_by_lift, 10),
     method = "graph",
     control = list(type = "items"),
     main = "Top 10 Rules Network Graph")

# Parallel coordinates plot
plot(head(rules_by_lift, 10),
     method = "paracoord",
     control = list(reorder = TRUE),
     main = "Top 10 Rules - Parallel Coordinates")

# -------------------------------------------------
# Step 8: Rule Quality Analysis
# -------------------------------------------------

# Calculate rule quality metrics
quality_metrics <- interestMeasure(rules,
                                   measure = c("coverage", "chiSquared", "cosine",
                                               "conviction", "leverage", "oddsRatio"),
                                   transactions = transactions)

# Add to rules
quality(rules) <- cbind(quality(rules), quality_metrics)

# Summary of rule quality
cat("\n=== RULE QUALITY SUMMARY ===\n")
summary(quality(rules))

# -------------------------------------------------
# Step 9: Generate Business Insights
# -------------------------------------------------

# Find rules with specific patterns
# Rules about fuel efficiency
mpg_rules <- subset(rules, lhs %pin% "MPG" | rhs %pin% "MPG")
cat("\n=== FUEL EFFICIENCY RULES ===\n")
if (length(mpg_rules) > 0) {
  inspect(head(sort(mpg_rules, by = "lift"), 5))
}

# Rules about specific brands
toyota_rules <- subset(rules, lhs %pin% "toyota" | rhs %pin% "toyota")
bmw_rules <- subset(rules, lhs %pin% "bmw" | rhs %pin% "bmw")

cat("\n=== TOYOTA-SPECIFIC RULES ===\n")
if (length(toyota_rules) > 0) {
  inspect(head(sort(toyota_rules, by = "lift"), 5))
}

cat("\n=== BMW-SPECIFIC RULES ===\n")
if (length(bmw_rules) > 0) {
  inspect(head(sort(bmw_rules, by = "lift"), 5))
}

# -------------------------------------------------
# Step 10: Export Results
# -------------------------------------------------

# Convert rules to dataframe for export
rules_df <- as(rules, "data.frame")

# Save all rules
write.csv(rules_df, "association_rules_all.csv", row.names = FALSE)

# Save top rules by lift
top_rules_df <- as(head(rules_by_lift, 50), "data.frame")
write.csv(top_rules_df, "association_rules_top50.csv", row.names = FALSE)

# Save transaction summary
transaction_summary <- data.frame(
  Total_Transactions = length(transactions),
  Total_Items = nitems(transactions),
  Avg_Items_per_Transaction = mean(size(transactions)),
  Min_Items = min(size(transactions)),
  Max_Items = max(size(transactions))
)
write.csv(transaction_summary, "transaction_summary.csv", row.names = FALSE)

# -------------------------------------------------
# Step 11: Statistical Summary
# -------------------------------------------------

# Calculate distribution of items
item_freq <- itemFrequency(transactions, type = "absolute")
item_freq_df <- data.frame(
  Item = names(item_freq),
  Frequency = item_freq,
  Percentage = item_freq / length(transactions) * 100
)
item_freq_df <- item_freq_df[order(-item_freq_df$Frequency), ]

cat("\n=== TOP 10 MOST FREQUENT ITEMS ===\n")
print(head(item_freq_df, 10))

# -------------------------------------------------
# Step 12: Interactive Visualization (Optional)
# -------------------------------------------------

# Uncomment for interactive visualization
# plot(rules, method = "interactive", shading = "lift")

# -------------------------------------------------
# OUTPUT SUMMARY
# -------------------------------------------------
cat("\n=== ASSOCIATION RULES ANALYSIS COMPLETE ===\n")
cat("Files saved:\n")
cat("1. association_rules_all.csv - All discovered rules\n")
cat("2. association_rules_top50.csv - Top 50 rules by lift\n")
cat("3. transaction_summary.csv - Transaction statistics\n")
cat("\nKey statistics:\n")
cat("Total transactions:", length(transactions), "\n")
cat("Total items:", nitems(transactions), "\n")
cat("Total rules discovered:", length(rules), "\n")
cat("Average items per transaction:", mean(size(transactions)), "\n")

# -------------------------------------------------
# Step 13: Generate Report-Ready Insights
# -------------------------------------------------

cat("\n=== BUSINESS INSIGHTS FROM ASSOCIATION RULES ===\n")

# Analyze top rules
if (length(rules) > 0) {
  top_rule <- rules_by_lift[1]
  cat("1. STRONGEST ASSOCIATION (Highest Lift):\n")
  cat("   Rule:", paste(labels(top_rule)), "\n")
  cat("   Lift:", quality(top_rule)$lift, "\n")
  cat("   Support:", quality(top_rule)$support, "\n")
  cat("   Confidence:", quality(top_rule)$confidence, "\n")
  cat("   Interpretation: This combination occurs", 
      round(quality(top_rule)$support * 100, 1), 
      "% more frequently than expected by chance.\n\n")
  
  # Find interesting patterns
  high_confidence_rules <- rules[quality(rules)$confidence > 0.8]
  if (length(high_confidence_rules) > 0) {
    cat("2. HIGH-CONFIDENCE PATTERNS (>80% confidence):\n")
    for (i in 1:min(3, length(high_confidence_rules))) {
      cat("   ", i, ". ", labels(high_confidence_rules[i]), "\n")
    }
  }
  
  # Market segment insights
  segment_rules <- subset(rules, rhs %pin% "label=")
  if (length(segment_rules) > 0) {
    cat("\n3. MARKET SEGMENT CHARACTERISTICS:\n")
    for (seg in c("Economy", "Performance", "Luxury")) {
      seg_rules <- subset(segment_rules, rhs %pin% seg)
      if (length(seg_rules) > 0) {
        top_seg_rule <- seg_rules[which.max(quality(seg_rules)$lift)]
        cat("   ", seg, " Cars: Typically have ", 
            gsub("\\{|\\)", "", gsub(".*\\{", "", labels(top_seg_rule))), "\n")
      }
    }
  }
}

# ==============================================
# ADDITIONAL ANALYSIS: RULES BY CLUSTER
# ==============================================

cat("\n=== ASSOCIATION RULES BY CLUSTER ===\n")

# Analyze rules for each cluster separately
for (cluster_name in unique(car_data$label)) {
  cat("\n---", cluster_name, "---\n")
  
  # Subset data for this cluster
  cluster_data <- car_data[car_data$label == cluster_name, cat_cols]
  cluster_trans <- as(cluster_data, "transactions")
  
  # Generate rules for this cluster
  cluster_rules <- apriori(cluster_trans,
                           parameter = list(support = 0.2,
                                            confidence = 0.6,
                                            minlen = 2))
  
  if (length(cluster_rules) > 0) {
    cluster_rules <- sort(cluster_rules, by = "lift")
    cat("Top rule for", cluster_name, ":\n")
    inspect(head(cluster_rules, 1))
    cat("Total rules found:", length(cluster_rules), "\n")
  } else {
    cat("No significant rules found (try lowering thresholds)\n")
  }
}