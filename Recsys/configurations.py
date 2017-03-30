# -*- coding: utf-8 -*-
"""
Created on Mon Mar 13 12:45:45 2017

@author: anbarasan.selvarasu
"""

class Configurations(object):
    
    def __init__(self):
        pass
    
    BASE_FILE = "C:/Users/anbarasan.selvarasu/Documents/Yelp/master.csv"
    
    DEFAULT_TRAINING_CONFIG = {'lambda' : 1
                              ,'k': 10 # Dimensionality of the latent feature space
                              ,'n_epochs': 100  # Number of epochs
                              ,'learning_rate':0.01  # Learning rate
                              ,'momentum': 0.9 # Momentum
                            }