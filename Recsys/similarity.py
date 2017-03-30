# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 13:28:12 2017

@author: anbarasan.selvarasu
"""
from utilitymatrix import UtilityMatrix
from configurations import Configurations

import numpy as np
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import mean_squared_error
from math import sqrt

class SimilarityBased(object):
    def __init__(self):
        pass
    
    def compute_user_similarity(self,data_matrix):
        user_similarity = pairwise_distances(data_matrix, metric='cosine')
        return user_similarity
        
    def compute_item_similarity(self,data_matrix):
        item_similarity = pairwise_distances(data_matrix.T, metric='cosine')
        return item_similarity
    
    def predict(self,ratings, similarity, between='user'):
        if between == 'user':
            mean_user_rating = ratings.mean(axis=1)
            #You use np.newaxis so that mean_user_rating has same format as ratings
            ratings_diff = ratings - mean_user_rating[:, np.newaxis] 
            pred = mean_user_rating[:, np.newaxis] + similarity.dot(ratings_diff) / np.array([np.abs(similarity).sum(axis=1)]).T
        elif between == 'item':
            pred = ratings.dot(similarity) / np.array([np.abs(similarity).sum(axis=1)])     
        return pred
        
    def rmse(self,prediction, ground_truth):
        prediction = prediction[ground_truth.nonzero()].flatten() 
        ground_truth = ground_truth[ground_truth.nonzero()].flatten()
        return sqrt(mean_squared_error(prediction, ground_truth))        
   
    
def main():
    print("Similarity Based Class")
    
    testsize = 0.25
    utility = UtilityMatrix(Configurations.BASE_FILE,'Phoenix',testsize)
    train_data,test_data = utility.runmethod()
    
    similarity = SimilarityBased()
    user_similarity = similarity.compute_user_similarity(train_data.values)#dataframe.values => passes an ndarray
    item_similarity = similarity.compute_item_similarity(train_data.values)
    
    user_prediction = similarity.predict(train_data.values, user_similarity,'user')
    item_prediction = similarity.predict(train_data.values, item_similarity,'item')
    
    rmse_user = similarity.rmse(user_prediction, test_data.values)    
    rmse_item = similarity.rmse(item_prediction, test_data.values)
    
    print ('User-based CF RMSE: ' + str(rmse_user))
    print ('Item-based CF RMSE: ' + str(rmse_item))
    
    
    
if __name__ == "__main__":
    main()