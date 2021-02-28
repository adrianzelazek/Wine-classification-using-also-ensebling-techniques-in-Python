# Wine classification in Python
### Author: Adrian Żelazek

In this project was used dataset concerning red and white wines. The dataset was created by merging two datasets. Target variables is wine_type where: 1 - red wine, 0 - white wine.
The main target of this project was to build, evaluate and compare models to predict class of wine (red / white). Then the best model was choosen based on statistics of classification model as well as visualization of ROC curves of each model on one plot with AUC score. In modelling section were used also ensembling techniques like Random Forest or boosting.

This project was developed for the purpose of practicing machine learning technology and data mining in Python.
Source of dataset: UCI Machine Learning Repository.

The modelling dataset (input dataset after modifications) is really small because has only 7922 observations and 13 variables include target variable (wine_type). By doing so, results of models may be overfitted because regardless of algorithms chosen, hiper parameters tunning or data engineering techniques implement, dataset large enought really important for models, good quality data is more important than algorithms.

The input dataset without data engineering techniques contained: 6497 observations as well as 13 variables. Input dataset after concatenation was presented on two different reports: Pyandas Profiling and Sweetviz.
Then many different data modifications processes were used: renaming columns, enumerate new variables, checking and changing data types, removing duplicates, missing variables, outliers detection by boxplots, Isolation Forest and Hampel method, checking of balance of target varaible, analysis of distribution of variables on 5 ways: histograms, Kolmogorov-Smirnov test, Shapiro-Wilk test, normal test from Scipy library, kurtosis nad skew.
Then data was visualized by scatter plots.

Before modelling were carried out dummy coding and then variables selection by: analysis of correlation (Pearson / Spearman), VIF, IF, Forward / Backward selection, TREE, RFE. Then was made oversampling by SMOTE method so as to make dataset balanced. Last thing before modelling was creation useful function to count quickly and easily: confusion matrix. classification report, ROC curve, comparision of statistics of models on test and train datasets to eventually detect overfitting. Because of small dataset, data was selected only by CORR and VIF methods.

Generally, 7 models were build (include bagging and boosting): Logistic Regression, KNN, SVM, Naive Bayes, Decision Tree, Random Forest and XGBoost. Each model were build after tunning of hyperparameters in classifier and in train test split so as to find both the best parameters of classifier and the best configuration of train test split.
In Logistic Regression tunning of hiper parameters was performed by GrichSearchCV, in rest models (KNN, SVM, Naive Bayes, Decision Tree, Random Forest and XGBoost) tunning of hiper parameters was performed by using loop which created different combinations of all choosen classifier parameters so as to achieve best AUC on test dataset and also simmilar AUC result on train and test datasets. Tunning of train test split was performed in each model by loop.

Each model was evaluated by: confusion matrix, classification report, ROC curve, AUC, Accuracy, Precision, Recall, F1, Gini. Moreover in each model was performed comparision of results on train and test dataset to eventually detect overfitting. Moveorver 2 plots for easy business interpretaion was plot from each model: PROFIT, LIFT.

Finally statistics of models were compared on one Data Frame and on one ROC plot. By doing so, can say that Random FOrest presents the highest Accuracy and the highest Precision together with XGBoost, nevertheless, althought XGBoost has slightly worse Accuracy than Random Forest, XGBoost has significantly higher results on Recall, so the best model from build models is XGBoost.

##### It is preferred to run/view the project via Jupyter Notebook (.ipynb) than via a browser (HTML).

### Programming language and platform
* Python - version : 3.7.4
* Jupyter Notebook

### Libraries
* Pandas version is 1.1.4
* Scipy version is 1.5.2
* Scikit-learn is 0.23.2
* Statsmodels is 0.12.1
* Numpy version is 1.20.1
* Matplotlib version is 3.3.3
* Seaborn version is 0.11.0
* XGBoost version is 1.3.3

### Algorithms
* Isolation Forest
* Hampel
* Kolmogorov-Smirnov
* Shapiro-Wilk
* normal test from Scipy
* dummy coding
* Pearson / Spearman corr
* VIF
* IV
* Forward/ Backwad
* TREE
* RFE

### Models built
* XGBoost
* Random Forest
* Decision Tree
* Naive Bayes
* SVM
* KNN
* Logistic Regression

### Methods of model evaluation
* Confusion matrix
* classification report
* Accuracy, Precision, Recall, F1
* AUC
* Gini
* ROC
* PROFIT
* LIFT
* comparision of statistics on train and test datasets
* comparision of models by DF with statistics and ROC with AUC for each model
