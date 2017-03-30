# -*- coding: utf-8 -*-
"""
Created on Sat Mar 18 16:36:40 2017

@author: anbarasan.selvarasu
"""

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
        self.n_epochs = cfg.DEFAULT_TRAINING_CONFIG['n_epochs']
        self.lmbda = cfg.DEFAULT_TRAINING_CONFIG['lambda']
        self.learning_rate = cfg.DEFAULT_TRAINING_CONFIG['learning_rate']
        self.latent_features = cfg.DEFAULT_TRAINING_CONFIG['k']
        self.momentum = cfg.DEFAULT_TRAINING_CONFIG['momentum']
        
    
    def create_index_matrix(self,train_data,validation_data, test_data):
        train_index = train_data.copy()
        train_index[train_index > 0] = 1
        train_index[train_index == 0] = 0
        
        # Index matrix for test data
        test_index = test_data.copy()
        test_index[test_index > 0] = 1
        test_index[test_index == 0] = 0
        
        # Index matrix for validation data
        validation_index = validation_data.copy()
        validation_index[validation_index > 0] = 1
        validation_index[validation_index == 0] = 0        
        
        return train_index,validation_index,test_index
    
    
    def compute_loss(self,index_matrix,ratings_matrix,user_params,item_params):
        '''
        This function computes the cost with regularization
        '''
        predicted_rating = self.prediction(user_params,item_params) # theta'.X
        sqrd_error = (ratings_matrix - predicted_rating)**2
        sqrd_error_ix_filtered = sqrd_error * index_matrix
        rss = np.sum(sqrd_error_ix_filtered.values)/2
        
        user_params_l2 =  np.dot(user_params.T,user_params)
        reg_user = (self.lmbda/2) * np.sum(user_params_l2)
        
        item_params_l2 = np.dot(item_params.T,item_params)
        reg_item = (self.lmbda/2) * np.sum(item_params_l2)

        cost = (rss + reg_user + reg_item)/np.sum(index_matrix.values)
        
        return cost
        
    def compute_loss_derivative(self,index_matrix,ratings_matrix,user_params,item_params):
        
        user_grad = np.dot((np.dot(user_params,item_params.T)*index_matrix - ratings_matrix),item_params )   + (self.lmbda * user_params)       
        user_grad = user_grad /np.sum(index_matrix.values)
         
        item_grad = np.dot((np.dot(user_params,item_params.T)*index_matrix - ratings_matrix).T,user_params)    + (self.lmbda * item_params)                
        item_grad = item_grad /np.sum(index_matrix.values)
        
        return user_grad,item_grad
        
        
    def prediction(self,P,Q):
        return np.dot(P,Q.T)
        
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
        
        m,n = train_data.shape
        
        lat_user = 3 * np.random.rand(self.latent_features,m) # Latent user feature matrix
        lat_item = 3 * np.random.rand(self.latent_features,n) # Latent item feature matrix
        
        #Only consider non-zero matrix 
        users,items = train_data.nonzero()      
        for epoch in range(self.n_epochs):
            for u, i in zip(users,items):
                error = train_data[u, i] - self.prediction(lat_user[:,u],lat_item[:,i])  # Calculate error for gradient
                lat_user[:,u] += self.alpha * ( error * lat_item[:,i] - self.lmbda * lat_user[:,u]) # Update latent user feature matrix
                lat_item[:,i] += self.alpha * ( error * lat_user[:,u] - self.lmbda * lat_item[:,i])  # Update latent item feature matrix
            train_rmse = self.rmse(train_index,train_data,lat_item,lat_user) # Calculate root mean squared error from train dataset
            test_rmse = self.rmse(test_index,test_data,lat_item,lat_user) # Calculate root mean squared error from test dataset
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
        return train_errors,test_errors
        
    def plot_training_curve(self,train_errors,validation_errors,test_errors):
        
        plt.plot(range(self.n_epochs), train_errors, marker='o', label='Training Data');
        plt.plot(range(self.n_epochs), test_errors, marker='p', label='validation Data');
        plt.plot(range(self.n_epochs), validation_errors, marker='v', label='Test Data');
        plt.title('SGD-WR Learning Curve')
        plt.xlabel('Number of Epochs');
        plt.ylabel('RMSE');
        plt.legend()
        plt.grid()
        plt.show()
        
    def train(self,train_data,test_data,validation_data,train_index,test_index,validation_index):
        
        train_errors= []
        validation_errors = []
        test_errors = []
        
        num_users,num_items = train_data.shape
    
        # Randomly initializae the trainable parameters        
        init_wt =  0.01
        user_params = np.random.normal(0., init_wt, size = (num_users,self.latent_features) )
        item_params = np.random.normal(0., init_wt, size = (num_items,self.latent_features))
        
        # Variables used for early stopping
        best_valid_loss = np.infty
        end_training = False
    
        # Initialize the momentum vector to all zeros
        user_delta = np.zeros((num_users,self.latent_features))
        item_delta = np.zeros(( num_items,self.latent_features))
    
        this_chunk_CE = 0.
        batch_count = 0
        num_epochs = self.n_epochs 
        # num_epochs = 1
        for epoch in range(1,num_epochs + 1):
            if end_training:
                break
            
            print()
            print('Epoch:%d' % epoch)
            
            # Compute loss derivative
            user_grad,item_grad = self.compute_loss_derivative(train_index,train_data,user_params,item_params)
                        
            
               
            # Measure loss function
            loss = self.compute_loss(train_index,train_data, user_params, item_params)
            #this_chunk_CE += cross_entropy
            
            print("Training Loss",loss)
            #print ('Batch {} Train CE {:1.3f}'.format(batch_count, this_chunk_CE ,config['show_training_CE_after']))
                
                
            # Update the momentum vector and user parameters
            user_delta = self.momentum * user_delta + user_grad
            user_params -= self.learning_rate * user_delta
            
            
            # Update the momentum vector and item parameters
            item_delta = self.momentum * item_delta + item_grad
            item_params -= self.learning_rate * item_delta
    
#==============================================================================
#             # Validate
#             if batch_count % config['show_validation_CE_after'] == 0:
#                 print('Running validation...')
#                 cross_entropy = model.evaluate(valid_inputs, valid_targets)
#                 print ('Validation cross-entropy: {:1.3f}'.format(cross_entropy))
# 
#             if cross_entropy > best_valid_CE:
#                 print ('Validation error increasing!  Training stopped.')
#                 end_training = True
#                 break
#==============================================================================
                
            train_rmse = self.rmse(train_index,train_data,item_params,user_params) # Calculate root mean squared error from train dataset
            validation_rmse = self.rmse(validation_index,validation_data,item_params,user_params) # Calculate root mean squared error from Validation dataset
            test_rmse = self.rmse(test_index,test_data,item_params,user_params) # Calculate root mean squared error from test dataset
            train_errors.append(train_rmse)
            validation_errors.append(validation_rmse)
            test_errors.append(test_rmse)     
            print("train_rmse",train_rmse)
            print("test_rmse",test_rmse)
            print("validation rmse",validation_rmse)
            
        
                
        print()
        print('Final training cross-entropy: %0.3f' %(train_errors[len(train_errors)-1]))
        print('Final validation cross-entropy: %0.3f'%(validation_errors[len(validation_errors)-1]))
        print ('Final test cross-entropy: %0.3f'%(test_errors[len(test_errors)-1]))
        self.plot_training_curve(train_errors,validation_errors,test_errors)
    
        
def main():
    print("Stochastic Gradient descent..")
    testsize = 0.25
    utility = UtilityMatrix(cfg.BASE_FILE,'Phoenix',testsize)
    train_data,validation_data,test_data = utility.runmethod()
    grad = GradientDescent()
    train_index,validation_index,test_index = grad.create_index_matrix(train_data, validation_data,test_data)
    #train_errors,test_errors = grad.train_model(train_data.values,test_data.values,train_index,test_index)
    #grad.plot_training_curve(train_errors,test_errors)
    grad.train(train_data,validation_data,test_data,train_index,validation_index,test_index)
    
    
    
#==============================================================================
#     num_users,num_items = train_data.shape
#     user_params = 3 * np.random.rand(num_users,5) # Latent user feature matrix
#     item_params = 3 * np.random.rand(num_items,5)
#     grad.compute_cost_func(train_index,train_data,user_params,item_params)
#     
#==============================================================================
    
    

if __name__ == "__main__":
    main()