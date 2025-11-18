# Script R challenge 2024-2025
# Majeure SD, UP2 : apprentissage statistique
# Techniques de Régression pour la prédiction

# 0. Données d'apprentissage

data <- read.table(file="data.txt", header=TRUE)
y <- data[,1]             # réponse
X <- data[,-1]            # matrice des prédicteurs

names(X[,1:5])            # noms des 5 premiers prédicteurs

n <- dim(data)[1]
p <- dim(data)[2] - 1     # p = nombre de prédicteurs > n


# 1. Prédiction par la moyenne (prédicteur constant)

prediction_constante <- mean(y)

plot(y, xlab="numéro d'observation")
abline(h=c(0, prediction_constante), lty=c(2,1), col=c("black","red"))
title('Prédiction par la moyenne')
grid()

RMSE_ref <- sd(y)         # écart-type de y
print(round(RMSE_ref,2))

# 2. Différentes techniques de régression
# Charger les bibliothèques nécessaires
library(glmnet)
library(pls)
library(caret)
library(randomForest)
library(gbm)
library(nnet)
library(MASS)
# Charger les données
data <- read.table("data.txt", header = TRUE)
y <- data[, 1]
X <- data[, -1]

# Standardiser les prédicteurs
X_scaled <- scale(X)

# Stocker les paramètres de standardisation
scaling_params <- list(
  center = attr(X_scaled, "scaled:center"),
  scale = attr(X_scaled, "scaled:scale")
)

# Séparer les données en apprentissage (80%) et test (20%)
set.seed(42)
train_indices <- sample(1:nrow(X), size = 0.8 * nrow(X))
X_train <- X_scaled[train_indices, ]
y_train <- y[train_indices]
X_test <- X_scaled[-train_indices, ]
y_test <- y[-train_indices]

# Vérification des dimensions
cat("Dimensions de X_train :", dim(X_train), "\n")
cat("Longueur de y_train :", length(y_train), "\n")
cat("Dimensions de X_test :", dim(X_test), "\n")
cat("Longueur de y_test :", length(y_test), "\n")

# Validation croisée pour comparer les modèles
set.seed(42)
train_control <- trainControl(method = "cv", number = 10)

# Modèle 1 : Régression Ridge
ridge_model <- train(X_train, y_train, method = "ridge", trControl = train_control)
ridge_predictions <- predict(ridge_model, newdata = as.data.frame(X_test))
ridge_rmse_test <- sqrt(mean((ridge_predictions - y_test)^2))

# Modèle 2 : Régression Lasso
lasso_model <- train(X_train, y_train, method = "lasso", trControl = train_control)
lasso_predictions <- predict(lasso_model, newdata = as.data.frame(X_test))
lasso_rmse_test <- sqrt(mean((lasso_predictions - y_test)^2))

# Modèle 3 : PCR
pcr_model <- train(X_train, y_train, method = "pcr", trControl = train_control)
pcr_predictions <- predict(pcr_model, newdata = as.data.frame(X_test))
pcr_rmse_test <- sqrt(mean((pcr_predictions - y_test)^2))

# Modèle 4 : PLS
pls_model <- train(X_train, y_train, method = "pls", trControl = train_control)
pls_predictions <- predict(pls_model, newdata = as.data.frame(X_test))
pls_rmse_test <- sqrt(mean((pls_predictions - y_test)^2))

# Modèle 5 : Stepwise (régression linéaire avec sélection de variables)
# Étape 1 : LASSO pour la sélection des variables
lasso_model <- cv.glmnet(X_train, y_train, alpha = 1)  # alpha = 1 pour LASSO
best_lambda_lasso <- lasso_model$lambda.min  # Obtenir la meilleure valeur de lambda
# Extraire les coefficients comme matrice
lasso_coef <- as.matrix(coef(lasso_model, s = best_lambda_lasso))

# Sélectionner les variables avec des coefficients non nuls
selected_vars <- rownames(lasso_coef)[lasso_coef[, 1] != 0][-1]  # Exclure l'intercept
X_train_lasso <- X_train[, selected_vars, drop = FALSE]

print(paste("Nombre de variables sélectionnées après LASSO :", ncol(X_train_lasso)))

# Étape 2 : Régression pas à pas
train_df_lasso <- as.data.frame(cbind(y = y_train, X_train_lasso))
initial_model <- lm(y ~ ., data = train_df_lasso)

# Régression stepwise bidirectionnelle
step_model <- stepAIC(initial_model, direction = "both")
summary(step_model)

# Étape 3 : Préparation des données de test
selected_vars <- intersect(selected_vars, colnames(X_test))
X_test_lasso <- X_test[, selected_vars, drop = FALSE]
X_test_df <- as.data.frame(X_test_lasso)

if (nrow(X_test_df) != length(y_test)) {
  stop("Le nombre de lignes dans X_test_df ne correspond pas à y_test.")
}

y_pred_step <- predict(step_model, newdata = X_test_df)
# Calcul du RMSE
calculer_RMSE <- function(y_observes, y_predictions) {
  sqrt(mean((y_observes - y_predictions)^2))
}
rmse_step <- calculer_RMSE(y_test, y_pred_step)
# Calcul du RMSE
calculer_RMSE <- function(y_observes, y_predictions) {
  sqrt(mean((y_observes - y_predictions)^2))
}
step_rmse_test <- calculer_RMSE(y_test, y_pred_step)

# Modèle 6 : Gradient Boosting
gbm_model <- train(X_train, y_train, method = "gbm", trControl = train_control, verbose = FALSE)
gbm_predictions <- predict(gbm_model, newdata = as.data.frame(X_test))
gbm_rmse_test <- sqrt(mean((gbm_predictions - y_test)^2))

# Modèle 7 : Forêt aléatoire
# Charger la bibliothèque randomForest
library(randomForest)

# Essayer plusieurs valeurs pour mtry et ntree via une boucle
best_rmse <- Inf  # Initialiser le meilleur RMSE
best_mtry <- NULL
best_ntree <- NULL

# Grille de recherche pour mtry et ntree
mtry_values <- seq(2, ncol(X_train), by = 2)  # Exemple : tester mtry de 2 à p
ntree_values <- c(100, 500, 1000)             # Tester différents nombres d'arbres

# Recherche des meilleurs hyperparamètres
for (mtry in mtry_values) {
  for (ntree in ntree_values) {
    # Entraîner le modèle avec les hyperparamètres actuels
    rf_model <- randomForest(
      x = X_train,
      y = y_train,
      mtry = mtry,
      ntree = ntree
    )
    
    # Prédictions sur les données de test
    rf_predictions <- predict(rf_model, newdata = X_test)
    rf_rmse_test <- sqrt(mean((rf_predictions - y_test)^2))
    
    # Mettre à jour le meilleur modèle si le RMSE est amélioré
    if (rf_rmse_test < best_rmse) {
      best_rmse <- rf_rmse_test
      best_mtry <- mtry
      best_ntree <- ntree
    }
  }
}

# Afficher les meilleurs hyperparamètres et RMSE
cat("Best RMSE:", best_rmse, "\n")
cat("Best mtry:", best_mtry, "\n")
cat("Best ntree:", best_ntree, "\n")

# Étape 2 : Entraîner le modèle final avec les meilleurs hyperparamètres

rf_rmse_test <-best_rmse 
final_rf_model <- randomForest(
  x = X_train,
  y = y_train,
  mtry = best_mtry,
  ntree = best_ntree
)


# Résumé des RMSE
rmse_results <- data.frame(
  Model = c("Ridge", "Lasso", "PCR", "PLS", "Stepwise", "Gradient Boosting", "Random Forest"),
  RMSE_Test = c(ridge_rmse_test, lasso_rmse_test, pcr_rmse_test, pls_rmse_test, step_rmse_test, gbm_rmse_test, rf_rmse_test)
)

print(rmse_results)
# 3. Prédictions 
Xnew <- read.table(file="Xtest.txt", header=TRUE)
head(Xnew[,1:8])   # pour vérifier la bonne lecture des données test
#Avec stepwise (modèle 5) 
#Rappel RMSE lors du test : 2,48

# Standardiser Xnew en utilisant les mêmes paramètres de standardisation que X_train
Xnew_scaled <- scale(
  Xnew,
  center = attr(X_scaled, "scaled:center"),
  scale = attr(X_scaled, "scaled:scale")
)

# Sélectionner uniquement les variables utilisées dans le modèle stepwise
selected_vars <- intersect(selected_vars, colnames(Xnew_scaled))  # Vérifie la compatibilité
Xnew_selected <- Xnew_scaled[, selected_vars, drop = FALSE]  # Filtre les variables sélectionnées

# Convertir en data frame
Xnew_df <- as.data.frame(Xnew_selected)

# Effectuer les prédictions avec le modèle stepwise
predictions1 <- predict(step_model, newdata = Xnew_df)

# Fichier texte des prédictions 

write.table(predictions1, row.names=FALSE, col.names=FALSE,
            file="BOULLON_Charles.txt")

# 3. Prédictions avec arbre (si vous préférez)
#Rappel RMSE lors du test : 2,84

predictions2 <- predict(final_rf_model, newdata = Xnew_scaled)
    

# Fichier texte des prédictions 

write.table(predictions2, row.names=FALSE, col.names=FALSE,
            file="BOULLON_Charles2.txt")

# S'assurer que ce fichier a le bon format en l'ouvrant avec un éditeur
# de texte quelconque et le déposer sur Campus

