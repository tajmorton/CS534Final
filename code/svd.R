
load_data <- function(filename) {
}

run_svd <- function(features) {
    return(prcomp(features))
}

create_data_partition <- function(examples, p = 0.5) {
    training_indices <- createDataPartition(examples,1,p=p)
    return(training_indices$Resample1)
}

train_model <-function(features, classes, training_indices, nfolds = 10, family="binomial") {
    return(cv.glmnet(features[training_indices,], classes[training_indices], nfolds=nfolds, family=family))
}

test_model <- function(model, features, classes, testing_indices, s = NULL) {
    pred_factors <- factor(predict(model, newx=features[testing_indices,], type="class", s=s), levels=levels(classes[testing_indices]))
    return(confusionMatrix(data=pred_factors, classes[testing_indices], positive=levels(classes[testing_indices])[-1]))
}

train_models_with_feature_range <- function(pc_ranges, ordered_feature_names, features, classes, training_indices, nfolds = 10, family="binomial") {
    results_table <- data.frame()
    for (pcs in pc_ranges) {
        print(pcs)
        model <- train_model(features[,ordered_feature_names[1:pcs], drop=FALSE], classes, training_indices, nfolds, family)
        model_results <- test_model(model, features[,ordered_feature_names[1:pcs]], classes, -training_indices, s=model$lambda.min)

        results_table <- rbind(results_table, c(model_results$byClass, model_results$overall["Accuracy"]))
        colnames(results_table) <- c(names(model_results$byClass), "Accuracy")
    }

    row.names(results_table) <- pc_ranges
    return(results_table)
}

train_models_with_pc_range <- function(pc_ranges, features, classes, training_indices, nfolds = 10, family="binomial") {
    results_table <- data.frame()
    for (pcs in pc_ranges) {
        print(pcs)
        model <- train_model(features[,1:pcs, drop=FALSE], classes, training_indices, nfolds, family)
        model_results <- test_model(model, features[,1:pcs], classes, -training_indices, s=model$lambda.min)

        results_table <- rbind(results_table, c(model_results$byClass, model_results$overall["Accuracy"]))
        colnames(results_table) <- c(names(model_results$byClass), "Accuracy")
    }

    row.names(results_table) <- pc_ranges
    return(results_table)
}

get_feature_weight_vector <- function(model, s = NULL) {
    if (is.null(s)) {
        s = model$lambda.1se
    }

    m_weights <- coef(model, s=s) # get sparse matrix with weights
    v_weights <- as.vector(m_weights)
    names(v_weights) <- dimnames(m_weights)[[1]]

    return(v_weights)
}

