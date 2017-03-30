# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:39:53 2017

@author: anbarasan.selvarasu
"""
import pandas as pd
import numpy as np
from sklearn import cross_validation as cv

from configurations import Configurations

class UtilityMatrix(object):
    
    def __init__(self,file_path,city,test_split):
        self.file_path = file_path
        self.city = city
        self.test_split = test_split
        
        
    def get_data(self):
        base_df = pd.read_csv(self.file_path, nrows = 30000,encoding = "latin-1")
        base_df = base_df[base_df['city']==self.city]
        utility_df = base_df[['user_id','business_id','stars_x']]
        return utility_df 
        
    def get_test_train_matrix(self,is_validation = False):
        utility_df = self.get_data()
        
        n_users = utility_df['user_id'].unique().shape[0]
        n_business = utility_df['business_id'].unique().shape[0]
        
        print("Num of users: %d | Num of Business: %d " %(n_users,n_business))        
        
        train_data, test_data = cv.train_test_split(utility_df, test_size= self.test_split)
        if is_validation:
            validation_data, test_data = cv.train_test_split(test_data, test_size= 0.5)
                
        user_id_ix = utility_df['user_id'].unique()
        business_id_ix = utility_df['business_id'].unique()
        
        train_data_matrix = pd.DataFrame(np.zeros((n_users, n_business))
                                        ,index = user_id_ix
                                        ,columns = business_id_ix)
        
        test_data_matrix = pd.DataFrame(np.zeros((n_users, n_business))
                                        ,index = user_id_ix
                                        ,columns = business_id_ix)
        
        for i in range(train_data.shape[0]):
            train_data_matrix.ix[train_data.iloc[i]['user_id'],train_data.iloc[i]['business_id']] = train_data.iloc[i]['stars_x']
        
        for i in range(test_data.shape[0]):
            test_data_matrix.ix[test_data.iloc[i]['user_id'],test_data.iloc[i]['business_id']] = test_data.iloc[i]['stars_x']
        
        if is_validation:                                        
            validation_data_matrix = test_data_matrix.copy()
            for i in range(validation_data.shape[0]):
                validation_data_matrix.ix[validation_data.iloc[i]['user_id'],validation_data.iloc[i]['business_id']] = validation_data.iloc[i]['stars_x']
            return train_data_matrix,validation_data_matrix,test_data_matrix
                                         
       
        return train_data_matrix,test_data_matrix
        
    

def main():
    print("hello")
    testsize = 0.25
    preparefile = UtilityMatrix(Configurations.BASE_FILE,'Phoenix',testsize)
    train,test = preparefile.get_test_train_matrix()
       
    

if __name__ == "__main__":
    main()
