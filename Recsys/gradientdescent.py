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
    
    
    
    def __init__(self):
        self.n_epochs = 100
        self.lmbda = 0.1
        self.learning_rate = 0.01
        self.latent_features = 10       
        
        
    
    def create_index_matrix(self,train_data,test_data):
        train_index = train_data.copy()
        train_index[train_index > 0] = 1
        train_index[train_index == 0] = 0
        
        # Index matrix for test data
        test_index = test_data.copy()
        test_index[test_index > 0] = 1
        test_index[test_index == 0] = 0
        
        return train_index,test_index
        
    def prediction(self,P,Q):
        return np.dot(P,Q.T)
        
    # Calculate the RMSE
    # Calculate the RMSE
    def rmse(self,index,data_matrix,lat_item,lat_user):
        predicted = self.prediction(lat_user,lat_item)
        error = data_matrix - predicted
        error_nonzero_ratings = index * error
        sqrd_err = error_nonzero_ratings ** 2
        sum_of_sqrd_err = np.sum(sqrd_err.values)
        return np.sqrt(sum_of_sqrd_err/np.sum(index.values))
        
    def train_model(self,train_data,test_data,train_index,test_index):
        train_errors = []
        test_errors = []        
        
        num_users,num_items = train_data.shape
        
        # Initialize Latent Factors
        lat_user = 3 * np.random.rand(num_users, self.latent_features) # Latent user feature matrix
        lat_item = 3 * np.random.rand(num_items, self.latent_features) # Latent item feature matrix
        
              
        
        #Only consider non-zero matrix 
        users,items = train_data.nonzero()      
        for epoch in range(self.n_epochs):
            for u, i in zip(users,items):
                error = train_data[u, i] - self.prediction(lat_user[u,:],lat_item[i,:])  # Calculate error for gradient

                # Update Latent Factors                
                lat_user[u,:] += self.learning_rate * \
                                ( error * lat_item[i,:] - self.lmbda * lat_user[u,:]) # Update latent user feature matrix
                lat_item[i,:] += self.learning_rate * \
                                ( error * lat_user[u,:] - self.lmbda * lat_item[i,:])  # Update latent item feature matrix
            
                
            train_rmse = self.rmse(train_index,train_data,lat_item,lat_user) # Calculate root mean squared error from train dataset
            test_rmse = self.rmse(test_index,test_data,lat_item,lat_user) # Calculate root mean squared error from test dataset
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
            
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
    testsize = 0.25
    utility = UtilityMatrix(cfg.BASE_FILE,'Phoenix',testsize)
    train_data,validation_data,test_data = utility.runmethod()
    grad = GradientDescent()
    train_index,test_index = grad.create_index_matrix(train_data, test_data)
    train_errors,test_errors = grad.train_model(train_data.values,test_data.values,train_index,test_index)
    grad.plot_training_curve(train_errors,test_errors)
    
    
    

if __name__ == "__main__":
    main()