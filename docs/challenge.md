

DS Exploration review:

After reviewing the DS exploration i noticed that there were dummies missing in the creation, i added them... but stilll after running the feature importance i noticed they are not within the most important features.

After checking the scoring values of the different models, i noticed that the logistic regression and the XGboost when balances and using the most representative features, both have similar values... being the F1 score value a bit better for the XGBoost, so thats the one selected.


model operationalize:


I created a pre_process class in order to have all main functions needed to process the features into the version that will be used for training/predicting

-preprocess: prepares the data to be trained or predicted. received a df or a df and the selected "target" value, will retrieve a tuple with both values or jusst the prepared DF

-fit: this takes the prcessed DF and trains the model in order to get the best features and the retrains it for better scoring

-predict: takes the preprocessed DF and only uses the top X features selected for the best model


model-test:

I had to perform some changes regarding the validation, as i aded a function that will always look for the best X features when fitted, this way you are not locked on the ones selected


fastAPI:

get/health: provides you with the information about if the model is trained and ready to provide predictions

get/train: trains the model with all data from the files

post/predict: if trained, provides with the prediction for the enteres values


