# -*- coding: utf-8 -*-
"""
Created on Sun Mar 19 09:15:09 2017

@author: anbarasan.selvarasu
"""
from utilitymatrix import UtilityMatrix
from configurations import Configurations as cfg

from numpy.linalg import solve
from sklearn.metrics import mean_squared_error
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
sns.set()   


class ExplicitMF():
    def __init__(self, 
                 ratings, 
                 n_factors=40, 
                 item_reg=0.0, 
                 user_reg=0.0,
                 verbose=False):
        """
        Train a matrix factorization model to predict empty 
        entries in a matrix. The terminology assumes a 
        ratings matrix which is ~ user x item
        
        Params
        ======
        ratings : (ndarray)
            User x Item matrix with corresponding ratings
        
        n_factors : (int)
            Number of latent factors to use in matrix 
            factorization model
        
        item_reg : (float)
            Regularization term for item latent factors
        
        user_reg : (float)
            Regularization term for user latent factors
        
        verbose : (bool)
            Whether or not to printout training progress
        """
        
        self.ratings = ratings
        self.n_users, self.n_items = ratings.shape
        self.n_factors = n_factors
        self.item_reg = item_reg
        self.user_reg = user_reg
        self._v = verbose

    def als_step(self,
                 latent_vectors,
                 fixed_vecs,
                 ratings,
                 lmbda,
                 type='user'):
        """
        One of the two ALS steps. Solve for the latent vectors
        specified by type.
        """
        if type == 'user':
            # Precompute
            yty = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(yty.shape[0]) * lmbda

            for u in range(latent_vectors.shape[0]):
                latent_vectors[u, :] = solve((yty + lambdaI), 
                                             ratings[u, :].dot(fixed_vecs))
        elif type == 'item':
            # Precompute
            xtx = fixed_vecs.T.dot(fixed_vecs)
            lambdaI = np.eye(xtx.shape[0]) * lmbda
            
            for i in range(latent_vectors.shape[0]):
                latent_vectors[i, :] = solve((xtx + lambdaI), 
                                             ratings[:, i].T.dot(fixed_vecs))
        return latent_vectors

    def train(self, n_iter=10):
        """ Train model for n_iter iterations from scratch."""
        # initialize latent vectors
        self.user_vecs = np.random.random((self.n_users, self.n_factors))
        self.item_vecs = np.random.random((self.n_items, self.n_factors))
        
        self.partial_train(n_iter)
    
    def partial_train(self, n_iter):
        """ 
        Train model for n_iter iterations. Can be 
        called multiple times for further training.
        """
        ctr = 1
        while ctr <= n_iter:
            if ctr % 10 == 0 and self._v:
                print('\tcurrent iteration: %d' %(ctr))
            self.user_vecs = self.als_step(self.user_vecs, 
                                           self.item_vecs, 
                                           self.ratings.values, 
                                           self.user_reg, 
                                           type='user')
            self.item_vecs = self.als_step(self.item_vecs, 
                                           self.user_vecs, 
                                           self.ratings.values, 
                                           self.item_reg, 
                                           type='item')
            ctr += 1
    
    def predict_all(self):
        """ Predict ratings for every user and item. """
        predictions = np.zeros((self.user_vecs.shape[0], 
                                self.item_vecs.shape[0]))
        for u in range(self.user_vecs.shape[0]):
            for i in range(self.item_vecs.shape[0]):
                predictions[u, i] = self.predict(u, i)
                
        return predictions
    def predict(self, u, i):
        """ Single user and item prediction. """
        return self.user_vecs[u, :].dot(self.item_vecs[i, :].T)
    
    def calculate_learning_curve(self, iter_array, test):
        """
        Keep track of MSE as a function of training iterations.
        
        Params
        ======
        iter_array : (list)
            List of numbers of iterations to train for each step of 
            the learning curve. e.g. [1, 5, 10, 20]
        test : (2D ndarray)
            Testing dataset (assumed to be user x item).
        
        The function creates two new class attributes:
        
        train_mse : (list)
            Training data MSE values for each value of iter_array
        test_mse : (list)
            Test data MSE values for each value of iter_array
        """
        iter_array.sort()
        self.train_mse =[]
        self.test_mse = []
        iter_diff = 0
        for (i, n_iter) in enumerate(iter_array):
            if self._v:
                print('Iteration: %d' %n_iter)
            if i == 0:
                self.train(n_iter - iter_diff)
            else:
                self.partial_train(n_iter - iter_diff)

            predictions = self.predict_all()

            self.train_mse += [self.get_mse(predictions, self.ratings.values)]
            self.test_mse += [self.get_mse(predictions, test.values)]
            if self._v:
                print ('Train mse:  ' + str(self.train_mse[-1]))
                print ('Test mse: ' + str(self.test_mse[-1]))
            iter_diff = n_iter
            
    def get_mse(self,pred, actual):
        # Ignore nonzero terms.
        pred = pred[actual.nonzero()].flatten()
        actual = actual[actual.nonzero()].flatten()
        return mean_squared_error(pred, actual)
        
def plot_learning_curve(iter_array,model):
        plt.plot(iter_array, model.train_mse, \
                 label='Training', linewidth=5)
        plt.plot(iter_array, model.test_mse, \
                 label='Test', linewidth=5)
    
    
        plt.xticks(fontsize=16);
        plt.yticks(fontsize=16);
        plt.xlabel('iterations', fontsize=30);
        plt.ylabel('MSE', fontsize=30);
        plt.legend(loc='best', fontsize=20);    

def main():
    testsize = 0.25
    utility = UtilityMatrix(cfg.BASE_FILE,'Phoenix',testsize)
    train_data,test_data = utility.get_test_train_matrix()
    MF_ALS = ExplicitMF(train_data, n_factors=20, user_reg=0.1, item_reg=0.1,verbose = True)
    
    iter_array = [1, 2, 5, 10,15,20, 25, 50, 100]
    MF_ALS.calculate_learning_curve(iter_array, test_data)
    plot_learning_curve(iter_array, MF_ALS)
    
if __name__ == "__main__":
    main()