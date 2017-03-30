# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 11:01:03 2017

@author: anbarasan.selvarasu
"""
from configurations import Configurations as cfg
from utilitymatrix import UtilityMatrix
import numpy as np
import scipy as sp
import matplotlib.pyplot as plt


class ALS_2(object):
    
    def __init__(self):
            self.n_epochs = 30
            self.lmbda = cfg.DEFAULT_TRAINING_CONFIG['lambda']
            self.learning_rate = cfg.DEFAULT_TRAINING_CONFIG['learning_rate']
            self.latent_features = cfg.DEFAULT_TRAINING_CONFIG['k']
            
    def create_index_matrix(self,train_data, test_data):
        train_index = train_data.copy()
        train_index[train_index > 0] = 1
        train_index[train_index == 0] = 0
        
        # Index matrix for test data
        test_index = test_data.copy()
        test_index[test_index > 0] = 1
        test_index[test_index == 0] = 0
        
#==============================================================================
#         # Index matrix for validation data
#         validation_index = validation_data.copy()
#         validation_index[validation_index > 0] = 1
#         validation_index[validation_index == 0] = 0        
#==============================================================================
        
        return train_index,test_index
        
    def prediction(self,P,Q):
        return np.dot(P,Q.T)

    # Calculate the RMSE
    def rmse(self,index,data_matrix,lat_item,lat_user):
        predicted = self.prediction(lat_user,lat_item)
        error = data_matrix - predicted
        error_nonzero_ratings = index * error
        sqrd_err = error_nonzero_ratings ** 2
        sum_of_sqrd_err = np.sum(sqrd_err.values)
        return np.sqrt(sum_of_sqrd_err/np.sum(index))
        
    def als_2(self,train_data,test_data,train_index,test_index):
        train_errors = []
        test_errors = []
        num_users,num_items = train_data.shape
        #train_data = np.asmatrix(train_data)
        
         # Randomly initializae the trainable parameters        
        init_wt =  0.01
        user_params = 3 * np.random.rand(num_users,self.latent_features) 
        item_params = 3 * np.random.rand(num_items,self.latent_features)
        #mean_list =np.nonzero(train_data).mean(axis = 0)
        item_params[:,0] = train_data[train_data!=0].mean(axis=0) # AVg rating for each movie
        item_params = np.nan_to_num(item_params)
        diagonal_matrix = np.eye(self.latent_features)
        
        #P = 3 * np.random.rand(k,m) # Latent user feature matrix
        #Q = 3 * np.random.rand(k,n) # Latent movie feature matrix
        #Q[0,:] = R[R != 0].mean(axis=0) # Avg. rating for each movie
        #E = np.eye(k) # (k x k)-dimensional idendity matrix
        
        # Repeat until convergence
        for epoch in range(self.n_epochs):
            #1.Fix item parameters  and estimate user parameters
            #2. Item parameters becomes constant and differentiate loss function w.r.t user parameters
            #3.Iterate for every user ,the ratings associated with that user . i.e ratings_of_user_u
            for i, Ii in enumerate(train_index):
                nui = np.count_nonzero(Ii) # Number of items user i has rated
                if (nui == 0): nui = 1 # Be aware of zero counts!
            
                # Least squares solution
                coress_item_params = np.dot(np.diag(Ii),item_params)
                yty = np.dot(coress_item_params.T, coress_item_params)
                reg = self.lmbda * nui * diagonal_matrix
                Ai =  yty+ reg
                Vi = np.dot(train_data.values[i],np.dot(np.diag(Ii),item_params))
                                
                user_params[i,:] = np.linalg.solve(Ai,Vi)
                
#==============================================================================
#                 YTY = item_params.T.dot(item_params)
#             lambdaI = np.eye(YTY.shape[0]) * lmbda
# 
#             for u in range(latent_vectors.shape[0]):
#                 latent_vectors[u, :] = solve((YTY + lambdaI), 
#                                              ratings[u, :].dot(fixed_vecs))
#==============================================================================
                
            # Fix user parameters and estimate item parameters
            for j, Ij in enumerate(train_index.T):
                nmj = np.count_nonzero(Ij) # Number of users that rated item j
                if (nmj == 0): nmj = 1 # Be aware of zero counts!
                
                # Least squares solution
                #Aj = np.dot(user_params.T, np.dot(np.diag(Ij), user_params.T)) + self.lmbda * nmj * diagonal_matrix
                #Vj = np.dot(user_params.T, np.dot(np.diag(Ij), train_data[:,j]))
                coress_user_params = np.dot(np.diag(Ij),user_params)
                xtx = np.dot(coress_user_params.T, coress_user_params)
                reg = self.lmbda * nmj * diagonal_matrix
                Aj =  xtx+ reg
                Vj = np.dot(train_data.values[:,j],np.dot(np.diag(Ij),user_params))                
                item_params[j,:] = np.linalg.solve(Aj,Vj)
            
            train_rmse = self.rmse(train_index,train_data,item_params,user_params)
            test_rmse = self.rmse(test_index,test_data,item_params,user_params)
            train_errors.append(train_rmse)
            test_errors.append(test_rmse)
            
            print("[Epoch %d/%d] train error: %f, test error: %f" \
            %(epoch+1, self.n_epochs, train_rmse, test_rmse))
            
        print ("Algorithm converged")
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
    print("ALS 2..")
    testsize = 0.25
    utility = UtilityMatrix(cfg.BASE_FILE,'Phoenix',testsize)
    train_data,test_data = utility.get_test_train_matrix()
    als = ALS_2()
    train_index,test_index = als.create_index_matrix(train_data,test_data)
    #train_errors,test_errors = grad.train_model(train_data.values,test_data.values,train_index,test_index)
    #grad.plot_training_curve(train_errors,test_errors)
    train_errors,test_errors = als.als_2(train_data,test_data,train_index.values,test_index.values)
    als.plot_training_curve(train_errors,test_errors)
    
if __name__ == "__main__":
    main()
        
    