
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
