# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:03:18 2017

@author: anbarasan.selvarasu
"""
from configurations import Configurations
from utilitymatrix import UtilityMatrix

import numpy as np

import scipy.sparse as sp
from scipy.sparse.linalg import svds

def decompose_data(train_data_matrix,test_data_matrix):
        
    #get SVD components from train matrix. Choose k.
    u, s, vt = svds(train_data_matrix, k = 20)
    s_diag_matrix=np.diag(s)
    
    X_pred = np.dot(np.dot(u, s_diag_matrix), vt)
    RMSE = rmse(X_pred, test_data_matrix)
    print('User-based CF MSE: ' + str(RMSE))
    
def rmse(self,prediction, ground_truth):
    prediction = prediction[ground_truth.nonzero()].flatten() 
    ground_truth = ground_truth[ground_truth.nonzero()].flatten()
    return sqrt(mean_squared_error(prediction, ground_truth)) 

def main():
    print("Matrix factorization")
    
    testsize = 0.25
    utility = UtilityMatrix(Configurations.BASE_FILE,'Phoenix',testsize)
    train_data,test_data = utility.runmethod()
    decompose_data(train_data.values,test_data.values)
    
if __name__ == "__main__":
    main()