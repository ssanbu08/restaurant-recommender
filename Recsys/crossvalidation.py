# -*- coding: utf-8 -*-
"""
Created on Mon Mar 20 07:06:19 2017

@author: anbarasan.selvarasu
"""
from gradientdescent_3 import GradientDescent
from utilitymatrix import UtilityMatrix
from configurations import Configurations as cfg

from sklearn import cross_validation as cv
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def plot_training_curve(errors):        
        plt.plot(range(100), errors, marker='o', label='validation Data');
        plt.title('SGD-WR Learning Curve')
        plt.xlabel('Number of Epochs');
        plt.ylabel('RMSE');
        plt.legend()
        plt.grid()
        plt.show()

def crossvalidate_coarse(train_data, validation_data):
        latent_factors = [5, 10, 20, 40, 80]
        regularizations = [0.001, 0.01, 0.1, 1.]
        regularizations.sort()
        
        best_params = {}
        best_params['n_factors'] = latent_factors[0]
        best_params['reg'] = regularizations[0]
        best_params['n_iter'] = 0
        best_params['train_mse'] = np.inf
        best_params['test_mse'] = np.inf
        best_params['model'] = None
        
        for fact in latent_factors:
            print ('Factors:',fact)
            for reg in regularizations:
                print('Regularization: ',reg)
                MF_SGD = GradientDescent(validation_data.values,learning_rate = 0.001, n_factors=fact, n_epochs = 100,\
                                    user_lat_reg=reg, item_lat_reg=reg, \
                                    user_bias_reg=reg, item_bias_reg=reg)
                train_index,validation_index = MF_SGD.create_index_matrix(train_data,validation_data)        
                train_errors,validation_errors = MF_SGD.train_model(train_data.values,validation_data.values\
                                                                    ,train_index,validation_index)
                #plot_training_curve(validation_errors)
                
                #MF_SGD.calculate_learning_curve(iter_array, test, learning_rate=0.001)
                                
                
                min_idx_valid = np.argmin(validation_errors)
                min_idx_train = np.argmin(train_errors)
                print("Factors : %d Regularization: %0.4f Training Error: %0.3f" \
                                    %(fact,reg,train_errors[min_idx_train]))
                print("Factors : %d Regularization: %0.4f validation Error: %0.3f" \
                        %(fact,reg,validation_errors[min_idx_valid]))
                
                print("Best Validation error so far: %0.4f " \
                        %(best_params['test_mse']))
                                                                
                # Update the best params if the new validation error is less than the error observed so far
                
                if validation_errors[min_idx_valid] < best_params['test_mse']:
                    best_params['n_factors'] = fact
                    best_params['reg'] = reg
                    #best_params['n_iter'] = iter_array[min_idx]
                    best_params['train_mse'] = train_errors[min_idx_train]
                    best_params['test_mse'] = validation_errors[min_idx_valid]
                    best_params['model'] = MF_SGD
                    print('New optimal hyperparameters')
                    print(pd.Series(best_params))
       
        print("Final Best parameters",pd.Series(best_params))
         

def main():
    print("cross-Validation for hyper parameter estimation...")
    testsize = 0.40
    utility = UtilityMatrix(cfg.BASE_FILE,'Phoenix',testsize)
    # STEP 1: Train & test split
    train_data,validation_data,test_data = utility.get_test_train_matrix(is_validation=True)
      
    # STEP 3: Train on training data and validate on validation_data
    crossvalidate_coarse(train_data,validation_data)
    
    
           
if __name__ == "__main__":
    main()
