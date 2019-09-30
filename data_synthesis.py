#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Andrew Floyd, Daniel Fuchs
Course: CS3001 - Dr. Fu
Data Science Competition Project
"""
import pandas as pd
from data_cleaning import *

#########################
# ENTRY LIST GENERATION #
#########################

def generate_user_list():
    t = pd.read_csv("train.csv")['userID']
    t = pd.DataFrame(t.unique())
    t.columns = ['userID']
    return t

def generate_restaurant_list():
    t = pd.read_csv("train.csv")['placeID']
    t = pd.DataFrame(t.unique())
    t.columns = ['placeID']
    return t

########################################
# TRAINING / TESTING DATA SEGMENTATION #
########################################

def generate_training_data_list():
    rating_table = clean_rating_table()
    t = pd.read_csv("train.csv")
    t.columns = ['revID', 'userID', 'placeID']
    t = t[['revID']]
    t = pd.merge(t, rating_table, how='left', on=['revID'])
    # t = t.drop(['revID'], axis=1)  # We don't need to have revID, unless we want to evaluate training's performance.
    return t

def generate_testing_data_list():
    rating_table = clean_rating_table()
    rating_table = rating_table.drop(['rating'], axis=1)
    t = pd.read_csv("test.csv")
    t.columns = ['revID', 'userID', 'placeID']
    t = t[['revID']]
    t = pd.merge(t, rating_table, how='left', on=['revID'])
    return t

def generate_testing_true_values():
    rating_table = clean_rating_table()
    t = pd.read_csv("test.csv")
    t.columns = ['revID', 'userID', 'placeID']
    t = t[['revID']]
    t = pd.merge(t, rating_table, how='left', on=['revID'])
    t = t.drop(['userID', 'placeID', 'service_rating', 'food_rating'], axis=1)
    return t

#####################
# RESTAURANT TABLES #
#####################

def synthesize_restaurant_profile():
    tables = [clean_loc_geo(), clean_loc_cuisine(), clean_loc_accepts(), clean_loc_parking(), clean_loc_hours()]
    restaurant_profile = pd.DataFrame(generate_restaurant_list())
    for table in tables:
        restaurant_profile = pd.merge(restaurant_profile, table, how='left', on=['placeID'])
    return standardize_restaurant_profile(restaurant_profile)

def standardize_restaurant_profile(table):
    for column in table.columns:
        table[column].fillna(table[column].mode()[0], inplace=True)
    return table

###############
# USER TABLES #
###############

def synthesize_user_profile():
    tables = [clean_user_profile(), clean_user_cuisine(), clean_user_payment()]
    user_profile = pd.DataFrame(generate_user_list())
    for table in tables:
        user_profile = pd.merge(user_profile, table, how='left', on=['userID'])
    return standardize_user_profile(user_profile)

def standardize_user_profile(table):
    for column in table.columns:
        table[column].fillna(table[column].mode()[0], inplace=True)
    return table

############################
# PRIMARY TABLE GENERATION #
############################

def synthesize_training_megatable():
    rating_table = generate_training_data_list()
    user_profile = synthesize_user_profile()
    rest_profile = synthesize_restaurant_profile()
    rest_reviews = pd.merge(rating_table, rest_profile, how='left', on=['placeID'])
    return pd.merge(rest_reviews, user_profile, how='left', on=['userID'])

def synthesize_testing_megatable():
    rating_table = generate_testing_data_list()
    user_profile = synthesize_user_profile()
    rest_profile = synthesize_restaurant_profile()
    rest_reviews = pd.merge(rating_table, rest_profile, how='left', on=['placeID'])
    return pd.merge(rest_reviews, user_profile, how='left', on=['userID'])
