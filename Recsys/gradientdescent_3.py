# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 18:16:03 2017

@author: anbarasan.selvarasu
"""

from utilitymatrix import UtilityMatrix
from configurations import Configurations as cfg

import numpy as np
import matplotlib.pyplot as plt

class GradientDescent(object):
    
    
    
    def __init__(self,train_data,learning_rate,n_factors,n_epochs,user_lat_reg,item_lat_reg \
                      ,user_bias_reg, item_bias_reg,_verbose = False):
        self.n_epochs = n_epochs
        self.lmbda = 0.1
        self.learning_rate = learning_rate
        self.n_factors = n_factors
        
        self.n_users = train_data.shape[0]
        self.n_items = train_data.shape[1]
        
         # Regularization strength
        self.user_lat_reg = user_lat_reg
        self.item_lat_reg = item_lat_reg
        self.user_bias_reg = user_bias_reg 
        self.item_bias_reg = item_bias_reg
        
        # Flag to control printing messages
        self._verbose = _verbose
        
        # Latent factors initialization
        
        self.lat_user = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_users, self.n_factors))
        self.lat_item = np.random.normal(scale=1./self.n_factors,\
                                          size=(self.n_items, self.n_factors))
        
        
        # Bias initialization
        self.user_bias = np.zeros(train_data.shape[0])
        self.item_bias = np.zeros(train_data.shape[1])
        self.global_bias = train_data[train_data != 0].mean()  
        
        
    
    def create_index_matrix(self,train_data,test_data):
        train_index = train_data.copy()
        train_index[train_index > 0] = 1
        train_index[train_index == 0] = 0
        
        # Index matrix for test data
        test_index = test_data.copy()
        test_index[test_index > 0] = 1
        test_index[test_index == 0] = 0
        
        return train_index,test_index
            
    def prediction(self,user_ix,item_ix):
        
        base_line = np.dot(self.lat_user[user_ix,:],self.lat_item[item_ix,:].T)
        predicted = base_line + self.user_bias[user_ix] + self.item_bias[item_ix] +\
                    self.global_bias 
        
        return predicted
    
    # Calculate the RMSE
    def rmse(self,index,data_matrix,lat_user,lat_item):
        
        predicted = np.dot(lat_user,lat_item.T)
        error = data_matrix - predicted
        error_nonzero_ratings = index * error
        sqrd_err = error_nonzero_ratings ** 2
        sum_of_sqrd_err = np.sum(sqrd_err.values)
        return np.sqrt(sum_of_sqrd_err/np.sum(index.values))
    
#==============================================================================
#     # For classification
#     def sigmoid(self,user_ix,item_ix):
#         z = np.dot(self.lat_user[user_ix,:],self.lat_item[item_ix,:].T)
#         base_line = 1/ (1+np.exp(-z))
#         return base_line
#     
#     # Calculate the ROC and other metrics
#     def roc(self,index,lat_user,lat_item):
#         z = np.dot(self.lat_user,self.lat_item.T)
#         predictions = 1/ (1+np.exp(-z))
#         return predictions       
#==============================================================================
    
        
    def train_model(self,train_data,test_data,\
                    train_index,test_index):
        train_errors = []
        test_errors = []        
        
               
        #Only consider non-zero matrix 
        users,items = train_data.nonzero()      
        for epoch in range(self.n_epochs):
            for user, item in zip(users,items):
                
                error = train_data[user, item] - self.prediction(user,item)  # Calculate error for gradient
                
                # Update Latent Factors                
                self.lat_user[user,:] += self.learning_rate * \
                                ( error * self.lat_item[item,:] - self.user_lat_reg * self.lat_user[user,:]) # Update latent user feature matrix
                self.lat_item[item,:] += self.learning_rate * \
                                ( error * self.lat_user[user,:] - self.item_lat_reg * self.lat_item[item,:])  # Update latent item feature matrix
            
                # Update Biases
                # Update biases
                self.user_bias[user] += self.learning_rate * \
                                (error - self.user_bias_reg * self.user_bias[user])
                self.item_bias[item] += self.learning_rate * \
                                (error - self.item_bias_reg * self.item_bias[item])
            
            
            train_rmse = self.rmse(train_index,train_data,self.lat_user,self.lat_item) # Calculate root mean squared error from train dataset
            test_rmse = self.rmse(test_index,test_data,self.lat_user,self.lat_item) # Calculate root mean squared error from test dataset
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)            
            
            if self._verbose:
                print("[Epoch %d/%d] train error: %0.3f, test error: %0.3f" \
            %(epoch+1, self.n_epochs, train_rmse, test_rmse))
        return train_errors,test_errors
        
    def plot_training_curve(self,train_errors,test_errors):
        
        plt.plot(range(self.n_epochs), train_errors, marker='o', label='Training Data');
        plt.plot(range(self.n_epochs), test_errors, marker='v', label='Test Data');
        plt.title('SGD-WR Learning Curve')
        plt.xlabel('Number of Epochs');
        plt.ylabel('RMSE');
        plt.legend()
        plt.grid()
        plt.show()
        
 
def main():
    print("Stochastic Gradinet descent..")
    testsize = 0.20
    utility = UtilityMatrix(cfg.BASE_FILE,'Phoenix',testsize)
    train_data,test_data = utility.get_test_train_matrix(is_validation=False)
    
    grad = GradientDescent(train_data = train_data.values
                            ,learning_rate = 0.001,n_factors = 10
                            ,n_epochs = 100, user_lat_reg = 0.1,item_lat_reg =  0.1
                            ,user_bias_reg = 0.1, item_bias_reg = 0.1
                            ,_verbose = True)
                            
                      
    train_index,test_index = grad.create_index_matrix(train_data,test_data)        
  
    train_errors,test_errors = grad.train_model(train_data.values,test_data.values\
                                                ,train_index,test_index)
    grad.plot_training_curve(train_errors,test_errors)
    
    # Binary classification problem
    #train_errors,test_errors = grad.train_model(train_data.values,test_data.values\
    #                                           ,train_index,test_index,False)
    
    

if __name__ == "__main__":
    main()