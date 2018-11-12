from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from sklearn.neural_network import MLPRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.svm import SVR


import pandas as pd
import numpy as np


YTEST = 614
TRAVELTIMEMIN = 0
TRAVELTIMEMAX = 7000
TRAINING_SET_PATH = "Given training set path"
TESTING_SET_PATH = "Given test set path"

def metricCalc(pred,y_test):
    
    print("\nAccuracy:")
    acc = r2_score(y_test, pred,multioutput='variance_weighted')
    print(acc)
    print("\nMSE:")
    mse = mean_squared_error(y_test, pred)
    print(mse)
    print("\nMAE:")
    mae = mean_absolute_error(y_test, pred)
    print(mae)
    print("\nMAPE:")
    
    errors = abs(pred - y_test)
    mape = np.mean(errors//y_test)
    
    return mape
    
def training():
    
    df1 = pd.read_csv(TRAINING_SET_PATH)
    df2 = pd.read_csv(TEST_SET_PATH)
        
    df1 = df1.drop(df1.columns[[0]],axis=1)
    df2 = df2.drop(df2.columns[[0]],axis=1)
    
    df1 = df1[df1['travel_time']>TRAVELTIMEMIN]
    df2 = df2[df2['travel_time']<TRAVELTIMEMAX]
    df1 = df1[df1['travel_time']>TRAVELTIMEMIN]
    df2 = df2[df2['travel_time']<TRAVELTIMEMAX]
    
    df1 = df1.reset_index()
    del df1['index']
    
    df2 = df2.reset_index()
    del df2['index']

    X_train = df1.iloc[:, :-2].values
    y_train = df1.iloc[:,YTEST].values
    X_train = pd.DataFrame(X_train)
    y_train = pd.DataFrame(y_train,columns=['travel_time'])
        
    dataset = [X_train,y_train]
    return dataset

def neuralNetwork(dataset):
    
    X_train, y_train = dataset
    clf = MLPRegressor(solver = 'lbfgs', activation = 'logistic',
                       alpha = 0.01)
    model = clf.fit(X_train,y_train)

    return model

def randomForest(dataset):
    
    X_train, y_train = dataset
    clf = RandomForestRegressor(n_estimators = 800, oob_score = 1,
                                n_jobs = -1,random_state =50,
                                max_features = "auto", 
                                min_samples_leaf = 50)
    model = clf.fit(X_train,y_train)
    
    return model

def gradBoost(dataset):
    
    X_train, y_train = dataset
    clf = GradientBoostingRegressor(loss = 'ls', learning_rate = 0.1,
                                    n_estimators = 400)
    model = clf.fit(X_train,y_train)

    return model
    
def svRegression(dataset):
    
    X_train, y_train = dataset
    clf = SVR(kernel = 'linear',C = 1e3, gamma = 0.1)
    model = clf.fit(X_train,y_train)
    
    return model

def prediction(model,X_test):
    pred = model.predict(X_test)
    return pred