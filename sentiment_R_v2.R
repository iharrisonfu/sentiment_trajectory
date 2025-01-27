library(lcmm)
library(readxl)
library("writexl")

data <- read_excel("C:/Users/Harrison/Documents/HKU/2024_07_24_heterogeneity_gpt/data/conv_stage_sentiment_score_avg_allIssue.xlsx")
head(data)

# Convert conversationId to a numeric factor
data$conversationId_numeric <- as.numeric(as.factor(data$conversationId))

# Convert stage_label to numeric
stage_mapping <- c("0~20" = 1, "20~40" = 2, "40~60" = 3, "60~80" = 4, "80~100" = 5)
data$stage_label_numeric <- stage_mapping[data$stage_label]

summary(data$stage_sentiment_score_avg)
summary(data$stage_label_numeric)
summary(data$conversationId_numeric)

# Calculate the quadratic term
# data$stage_label_numeric2 <- data$stage_label_numeric^2

model1 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, ng=1, data=data, subject = 'conversationId_numeric', link = "beta")

model2 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, mixture = ~ stage_label_numeric + I(stage_label_numeric^2), ng=2, data=data, subject = 'conversationId_numeric', link = "beta", B = model1)

model3 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, mixture = ~ stage_label_numeric + I(stage_label_numeric^2), ng=3, data=data, subject = 'conversationId_numeric', link = "beta", B = model1)

model4 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, mixture = ~ stage_label_numeric + I(stage_label_numeric^2), ng=4, data=data, subject = 'conversationId_numeric', link = "beta", B = model1)

model5 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, mixture = ~ stage_label_numeric + I(stage_label_numeric^2), ng=5, data=data, subject = 'conversationId_numeric', link = "beta", B = model1)

model6 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, mixture = ~ stage_label_numeric + I(stage_label_numeric^2), ng=6, data=data, subject = 'conversationId_numeric', link = "beta", B = model1)

#model7 <- lcmm(fixed = stage_sentiment_score_avg ~ stage_label_numeric + I(stage_label_numeric^2), random = ~ 1, mixture = ~ stage_label_numeric + I(stage_label_numeric^2), ng=7, data=data, subject = 'conversationId_numeric', link = "beta", B = model1)

# prediction and classification
pred3 <- predictY(model3, data, var.time = "stage_label_numeric")
#plot(pred3, col=c("red", "navy", "yellow"), lty=1, lwd=5, bty="l", type="l", shades = TRUE, ylab="pred", legend=NULL, main="Predicted trajectories", ylim=c(-1,0))
stage_label_numeric <- data$stage_label_numeric
pred_values <- as.data.frame(pred3$pred)

# Combine into a dataframe
pred_df <- data.frame(stage_label_numeric, pred_values)

# Sort the dataframe based on stage_label_numeric
pred_df_sorted <- pred_df[order(pred_df$stage_label_numeric), ]

# Extract the sorted predictions
sorted_stage_label_numeric <- pred_df_sorted$stage_label_numeric
sorted_pred_values <- pred_df_sorted[, -1]  # Exclude the stage_label_numeric column

# alternative way to draw the lines
plot(sorted_stage_label_numeric, sorted_pred_values[, 1], type="l", col="navy", lty=1, lwd=2, ylab="Sentiment Score", ylim=c(-1, 0), xlab="Epoch", cex.lab = 0.8, cex.axis = 0.7, cex.main = 0.8)
# , main="Latent Class Trajectories of Sentiment Scores"
lines(sorted_stage_label_numeric, sorted_pred_values[, 2], col="red", lty=1, lwd=2)
lines(sorted_stage_label_numeric, sorted_pred_values[, 3], col="orange", lty=1, lwd=2)
legend(x="topleft", legend=c("class1", "class2", "class3"), col=c("navy", "red", "orange"), lty=c(1,1,1), lwd=c(2,2,2), cex=0.8)

# get posterior probabilities
post_probs <- postprob(model3)

# Assign class membership based on the highest posterior probability
class_membership <- predictClass(model3, data)

# value_counts() function in python
class_counts <- class_membership %>% count(class)

# export class_membership to excel
write_xlsx(class_membership, "C:/Users/Harrison/Documents/HKU/2024_07_24_heterogeneity_gpt/data/class_membership_20250125.xlsx")

write_xlsx(data, "C:/Users/Harrison/Documents/HKU/2024_07_24_heterogeneity_gpt/data/conv_stage_avg_sentiment_score_allIssue_r.xlsx")
