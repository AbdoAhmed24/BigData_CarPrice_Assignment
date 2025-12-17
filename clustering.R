# ==============================================
# PHASE 3: CLUSTERING (K-MEANS)
# ==============================================

#install.packages("factoextra")
#install.packages("fmsb")

# Load necessary libraries
library(ggplot2)
library(factoextra)  # For clustering visualization

# -------------------------------------------------
# Step 1: Load and Prepare Dataset
# -------------------------------------------------
car_data <- read.csv("C:/hassan/cess/semester9/big data/project/BigData_CarPrice_Assignment-main/BigData_CarPrice_Assignment-main/car_prices_ml_ready.csv")
# View structure
str(car_data)
summary(car_data)

# Remove price column for clustering (we want to cluster based on features, not price)
clustering_data <- car_data[, !names(car_data) %in% c("price")]

# Check for missing values
sum(is.na(clustering_data))

# -------------------------------------------------
# Step 2: Standardize the Data
# -------------------------------------------------
# K-means is sensitive to scale differences
scaled_data <- scale(clustering_data)

# -------------------------------------------------
# Step 3: Determine Optimal Number of Clusters (Elbow Method)
# -------------------------------------------------
# Function to calculate total within-cluster sum of squares
wss <- numeric(15)

for (k in 1:15) {
  set.seed(123)
  kmeans_result <- kmeans(scaled_data, centers = k, nstart = 10)
  wss[k] <- kmeans_result$tot.withinss
}

# Plot elbow curve
elbow_data <- data.frame(Clusters = 1:15, WSS = wss)

ggplot(elbow_data, aes(x = Clusters, y = WSS)) +
  geom_line(color = "blue", size = 1.2) +
  geom_point(color = "red", size = 3) +
  labs(title = "Elbow Method for Optimal K",
       x = "Number of Clusters (K)",
       y = "Within-Cluster Sum of Squares (WSS)") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Alternative: Using fviz_nbclust (requires factoextra)
# fviz_nbclust(scaled_data, kmeans, method = "wss") +
#   geom_vline(xintercept = 3, linetype = 2) +
#   labs(title = "Elbow Method")

# -------------------------------------------------
# Step 4: Apply K-Means Clustering
# -------------------------------------------------
# Based on elbow plot, choose optimal k (let's say k=3 for example)
k <- 3
set.seed(123)  # For reproducibility
kmeans_model <- kmeans(scaled_data, centers = k, nstart = 10)

# -------------------------------------------------
# Step 5: Analyze Clusters
# -------------------------------------------------
# Add cluster labels to original data
car_data$cluster <- as.factor(kmeans_model$cluster)

# View cluster sizes
cluster_sizes <- table(car_data$cluster)
print("Cluster Sizes:")
print(cluster_sizes)

# View cluster centers (scaled)
print("Cluster Centers (Scaled):")
print(kmeans_model$centers)

# Convert scaled centers back to original scale for interpretation
original_centers <- t(apply(kmeans_model$centers, 1, 
                            function(r) r * attr(scaled_data, 'scaled:scale') + 
                              attr(scaled_data, 'scaled:center')))

print("Cluster Centers (Original Scale):")
print(original_centers)

# -------------------------------------------------
# Step 6: Visualize Clusters
# -------------------------------------------------
# Method 1: PCA for 2D visualization
pca_result <- prcomp(scaled_data)
pca_df <- data.frame(PC1 = pca_result$x[,1], 
                     PC2 = pca_result$x[,2], 
                     Cluster = car_data$cluster)

# PCA Scatter plot
ggplot(pca_df, aes(x = PC1, y = PC2, color = Cluster)) +
  geom_point(alpha = 0.7, size = 2) +
  stat_ellipse(level = 0.95) +
  labs(title = "K-Means Clustering Results (PCA Reduced)",
       x = paste("PC1 (", round(summary(pca_result)$importance[2,1]*100, 1), "%)", sep=""),
       y = paste("PC2 (", round(summary(pca_result)$importance[2,2]*100, 1), "%)", sep="")) +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# Method 2: Visualize specific feature pairs
ggplot(car_data, aes(x = horsepower, y = price, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Clusters: Horsepower vs Price",
       x = "Horsepower",
       y = "Price") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

ggplot(car_data, aes(x = enginesize, y = price, color = cluster)) +
  geom_point(alpha = 0.7) +
  labs(title = "Clusters: Engine Size vs Price",
       x = "Engine Size",
       y = "Price") +
  theme_minimal() +
  theme(plot.title = element_text(hjust = 0.5))

# -------------------------------------------------
# Step 7: Interpret Clusters
# -------------------------------------------------
# Calculate mean values for each cluster
cluster_summary <- aggregate(. ~ cluster, data = car_data, FUN = mean)

print("Cluster Summary (Mean Values):")
print(cluster_summary[, c("cluster", "horsepower", "enginesize", "curbweight", 
                          "citympg", "highwaympg", "price")])

# Label clusters based on characteristics
cluster_labels <- data.frame(
  cluster = 1:k,
  label = c("Economy Cars", "Performance Cars", "Luxury Cars")  # Adjust based on your findings
)

# Add labels to data
car_data <- merge(car_data, cluster_labels, by = "cluster")

# View distribution of brands across clusters
if (any(grepl("brand", names(car_data)))) {
  brand_cols <- grep("brand", names(car_data), value = TRUE)
  brand_cluster <- car_data[, c("cluster", brand_cols)]
  
  # Calculate percentage of each brand in each cluster
  brand_summary <- aggregate(. ~ cluster, data = brand_cluster, FUN = sum)
  print("Brand Distribution Across Clusters:")
  print(brand_summary)
}

# -------------------------------------------------
# Step 8: Save Results
# -------------------------------------------------
# Save clustered data
write.csv(car_data, "car_data_with_clusters.csv", row.names = FALSE)

# Save cluster centers
write.csv(original_centers, "cluster_centers.csv", row.names = FALSE)

# -------------------------------------------------
# Step 9: Visualization of Cluster Characteristics
# -------------------------------------------------
# Create radar chart data (requires fmsb package)
# install.packages("fmsb")
library(fmsb)

# Select key features for radar chart
key_features <- c("horsepower", "enginesize", "curbweight", "citympg", "price")
radar_data <- cluster_summary[, c("cluster", key_features)]

# Normalize for radar chart
max_values <- apply(radar_data[, -1], 2, max)
min_values <- apply(radar_data[, -1], 2, min)
radar_scaled <- as.data.frame(scale(radar_data[, -1], 
                                    center = min_values, 
                                    scale = max_values - min_values))
radar_scaled <- rbind(rep(1, ncol(radar_scaled)), 
                      rep(0, ncol(radar_scaled)), 
                      radar_scaled)

# Create radar chart
colors <- c("red", "blue", "green", "orange", "purple")[1:k]
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
           title = "Cluster Characteristics Radar Chart")

legend(x = 1.2, y = 1, 
       legend = paste("Cluster", 1:k, ":", cluster_labels$label), 
       bty = "n", pch = 20, col = colors, cex = 0.8)

# -------------------------------------------------
# Step 10: Statistical Summary
# -------------------------------------------------
# ANOVA test to check if clusters are significantly different for key features
anova_results <- list()
for (feature in c("horsepower", "enginesize", "price")) {
  anova_test <- aov(car_data[[feature]] ~ car_data$cluster)
  anova_results[[feature]] <- summary(anova_test)
}

print("ANOVA Results (Checking Cluster Differences):")
print(anova_results)

# ==============================================
# OUTPUT SUMMARY
# ==============================================
cat("\n=== CLUSTERING ANALYSIS SUMMARY ===\n")
cat("Number of clusters used:", k, "\n")
cat("Cluster sizes:\n")
print(cluster_sizes)
cat("\nCluster interpretation:\n")
print(cluster_labels)
cat("\nFiles saved:\n")
cat("1. car_data_with_clusters.csv - Dataset with cluster labels\n")
cat("2. cluster_centers.csv - Cluster centers in original scale\n")