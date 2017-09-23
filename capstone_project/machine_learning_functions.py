import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import learning_curve

from matplotlib import colors
import matplotlib.pyplot as plt

#%%
def add_features(daily_return_data, features, symbols, num_of_delays=0):
    """ Add additional features with past few days stock data specified by num_of_delays """   
    if num_of_delays==0:
        return features
    
    features = features.copy()
    
    for symbol in symbols:
        for i in range(0,num_of_delays):
            features[symbol+"(-"+str(i+1) +")"] = daily_return_data[symbol].shift(i+1)
    
    # Fill NaN values
    features = features.fillna(method='bfill')        
    
    return features

def pca_transform(X_train, X_test, n_components=7, random_state=7):
    """Apply PCA on the input features"""        
    pca = PCA(n_components=n_components,random_state=random_state)
    pca.fit(X_train)
    
    #print ("{}".format(n_components))
    #print ('Accumulative Variance Explained by Each of the Components')
    #print (pca.explained_variance_ratio_.cumsum()) 
        
    return pca.transform(X_train), pca.transform(X_test)

def get_train_test_data(features, output_label, split_date = '2015-01-01', isTensor = False):
    """Split the data set into training set and test set """
    start = features.index[0]
    end = features.index[-1]
    
    X_train = features.loc[start:split_date]
    X_test = features.loc[split_date:end]
    
    if isTensor:
        y_train = output_label.loc[start:split_date]
        y_test = output_label.loc[split_date:end]  
    else:
        y_train = output_label.loc[start:split_date,'Return Positive']
        y_test = output_label.loc[split_date:end,'Return Positive']   
    
    return X_train, y_train, X_test, y_test



#%%

def train_decision_tree(X_train,y_train,X_test,y_test,random_state=33):
    """Train a decision tree classifier using default setting"""
    dtree = DecisionTreeClassifier(random_state=random_state)
    dtree.fit(X_train,y_train)
    predictions = dtree.predict(X_test)    

    return dtree, accuracy_score(y_test,predictions), f1_score(y_test,predictions)

def train_logistic_regression(X_train,y_train,X_test,y_test,random_state=10):
    """Train a logistic regression classifier using default setting"""
    logmodel = LogisticRegression(random_state=random_state)
    
    logmodel.fit(X_train,y_train)
    predictions = logmodel.predict(X_test)    

    return logmodel, accuracy_score(y_test,predictions), f1_score(y_test,predictions)

def train_svm(X_train,y_train,X_test,y_test,random_state=30):
    """Train a SVM using default setting"""    
    svc = SVC(random_state=random_state)
    svc.fit(X_train,y_train)
    predictions = svc.predict(X_test)    

    return svc, accuracy_score(y_test,predictions), f1_score(y_test,predictions)

#%%
def grid_search(estimator, parameters, scorer, X_train, y_train, X_test, y_test):
    """Find the parameter values for an estimator by using Grid Search method"""
    # Perform grid search on the classifier using 'scorer' as the scoring method
    grid_obj = GridSearchCV(estimator,param_grid=parameters,scoring=scorer)

    # Fit the grid search object to the training data and find the optimal parameters
    grid_fit = grid_obj.fit(X_train, y_train)

    # Get the best estimator
    best_estimator = grid_fit.best_estimator_

    # Make predictions using the optimized model
    best_predictions = best_estimator.predict(X_test)
    
    print(grid_fit.best_params_)
    
    return best_estimator, accuracy_score(y_test,best_predictions), f1_score(y_test,best_predictions)

#%%

ddl_heat = ['#DBDBDB','#DCD5CC','#DCCEBE','#DDC8AF','#DEC2A0','#DEBB91',\
            '#DFB583','#DFAE74','#E0A865','#E1A256','#E19B48','#E29539']
ddlheatmap = colors.ListedColormap(ddl_heat)

def plot_classification_report(cr, title=None, cmap=ddlheatmap):
    """
    Plot the classification report heatmap
    This function is based on the code published on https://goo.gl/AvmW7v 
    """
    
    title = title or 'Classification report'
    lines = cr.split('\n')
    classes = []
    matrix = []

    for line in lines[2:(len(lines)-3)]:
        s = line.split()
        classes.append(s[0])
        value = [float(x) for x in s[1: len(s) - 1]]
        matrix.append(value)

    fig, ax = plt.subplots(1)

    for column in range(len(matrix)+1):
        for row in range(len(classes)):
            txt = matrix[row][column]
            ax.text(column,row,matrix[row][column],va='center',ha='center')

    fig = plt.imshow(matrix, interpolation='nearest', cmap=cmap)
    plt.title(title)
    plt.colorbar()
    x_tick_marks = np.arange(len(classes)+1)
    y_tick_marks = np.arange(len(classes))
    plt.xticks(x_tick_marks, ['precision', 'recall', 'f1-score'], rotation=45)
    plt.yticks(y_tick_marks, classes)
    plt.ylabel('Classes')
    plt.xlabel('Measures')
    plt.show()

#%%

def plot_learning_curve(estimator, title, X, y, ylim=None, cv=None,
                        n_jobs=1, train_sizes=np.linspace(.1, 1.0, 5)):
    """
    Generate a simple plot of the test and training learning curve.
    This function is published on sklearn website @ 
	http://scikit-learn.org/stable/_downloads/plot_learning_curve.py

    Parameters
    ----------
    estimator : object type that implements the "fit" and "predict" methods
        An object of that type which is cloned for each validation.

    title : string
        Title for the chart.

    X : array-like, shape (n_samples, n_features)
        Training vector, where n_samples is the number of samples and
        n_features is the number of features.

    y : array-like, shape (n_samples) or (n_samples, n_features), optional
        Target relative to X for classification or regression;
        None for unsupervised learning.

    ylim : tuple, shape (ymin, ymax), optional
        Defines minimum and maximum yvalues plotted.

    cv : int, cross-validation generator or an iterable, optional
        Determines the cross-validation splitting strategy.
        Possible inputs for cv are:
          - None, to use the default 3-fold cross-validation,
          - integer, to specify the number of folds.
          - An object to be used as a cross-validation generator.
          - An iterable yielding train/test splits.

        For integer/None inputs, if ``y`` is binary or multiclass,
        :class:`StratifiedKFold` used. If the estimator is not a classifier
        or if ``y`` is neither binary nor multiclass, :class:`KFold` is used.

        Refer :ref:`User Guide <cross_validation>` for the various
        cross-validators that can be used here.

    n_jobs : integer, optional
        Number of jobs to run in parallel (default 1).
    """
    plt.figure()
    plt.title(title)
    if ylim is not None:
        plt.ylim(*ylim)
    plt.xlabel("Training examples")
    plt.ylabel("Score")
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)
    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1,
                     color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")
    plt.show()
    return plt





