#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Andrew Floyd, Daniel Fuchs
Course: CS3001 - Dr. Fu
Data Science Competition Project
"""
from numpy import cos
import random as rn
import numpy as np

###########################
# FEATURE NAME MANAGEMENT #
###########################

def mass_feature_rename(table):
    translator = {'revID': 'revID',
                  'userID': 'userID',
                  'placeID': 'placeID',
                  'rating': 'RATING',
                  'food_rating': 'food_rating',
                  'service_rating': 'service_rating',
                  'alcohol': 'R_alcohol',
                  'smoking_area': 'R_smoking',
                  'accessibility': 'R_accessibility',
                  'price': 'R_price',
                  'franchise': 'R_franchise',
                  'other_services': 'R_services',
                  'Rformal_dress': 'R_formal_dress',
                  'Rquiet': 'R_quiet',
                  'open_area': 'R_open_area',
                  'Rlatitude': 'R_latitude',
                  'Rlongitude': 'R_longitude',
                  'Rcuisine': 'R_cuisine',
                  'accepts_cash': 'R_cash',
                  'accepts_visa': 'R_visa',
                  'accepts_mc_ec': 'R_mc_ec',
                  'accepts_am_exp': 'R_am_exp',
                  'accepts_debit': 'R_debit',
                  'accepts_check': 'R_check',
                  'free_parking': 'R_park_free',
                  'paid_parking': 'R_park_paid',
                  'no_parking': 'R_park_none',
                  'weekdays': 'R_weekday_hrs',
                  'sat_hours': 'R_sat_hrs',
                  'sun_hours': 'R_sun_hrs',
                  'latitude': 'U_latitude',
                  'longitude': 'U_longitude',
                  'smoker': 'U_smoking',
                  'drink_level': 'U_alcohol',
                  'transport': 'U_transport',
                  'interest': 'U_interest',
                  'personality': 'U_personality',
                  'activity': 'U_activity',
                  'weight': 'U_weight',
                  'budget': 'U_budget',
                  'formal_dress': 'U_formal_dress',
                  'quiet': 'U_quiet',
                  'married': 'U_married',
                  'age': 'U_age',
                  'cuisine': 'U_cuisine',
                  'uses_cash': 'U_cash',
                  'uses_visa': 'U_visa',
                  'uses_mc_ec': 'U_mc_ec',
                  'uses_am_exp': 'U_am_exp',
                  'uses_debit': 'U_debit'}
    for column in table.columns:
        if column not in translator:
            continue
        translation = translator[column]
        if translation == column:
            continue
        else:
            table[translation] = table[column]
            table = table.drop([column], axis=1)
    return table

def extract_features(table):
    non_features = ['revID', 'userID', 'placeID', 'RATING']
    features = []
    for column in table.columns:
        if column not in non_features:
            features.append(column)
    return table[features]

def quantify_features(table, remove_bool=True, remove_float=False):
    boolean_features = []
    continuous_features = []
    features = table.columns
    for x in table.dtypes:
        boolean_features.append(x == 'bool')
        continuous_features.append(x == 'float64')
    for i in range(len(features)):
        if remove_bool and boolean_features[i]:
            table.loc[table[features[i]] == True, features[i]] = 1
            table.loc[table[features[i]] == False, features[i]] = 0
        if remove_float and continuous_features[i]:
            table[features[i]] = table[features[i]].astype(int)
    return table

def finalize_feature_selections(table):
    # Important: ['RATING', 'revID', 'food_rating', 'service_rating', 'R_open_area', 'R_accessibility', 'match_quiet']
    critical_features = ['RATING', 'revID', 'food_rating', 'service_rating', 'R_accessibility', 'R_franchise',
                         'R_services', 'R_open_area', 'U_activity', 'U_married', 'U_age', 'proximity', 'days_open',
                         'smoking_score', 'U_smoker', 'alcohol_score', 'match_quiet', 'match_dress', 'U_personality',
                         'cuisine_score']

    for column in table.columns:
        if column not in critical_features:
            table = table.drop([column], axis=1)
    return table

#############################
# FEATURE: PAYMENT MATCHING #
#############################

def feature_payment_score(table, remove=True):
    r_fs = ['R_cash', 'R_visa', 'R_mc_ec', 'R_am_exp', 'R_debit']
    u_fs = ['U_cash', 'U_visa', 'U_mc_ec', 'U_am_exp', 'U_debit']
    n_fs = ['cash', 'visa', 'mc_ec', 'am_exp', 'debit']
    for i in range(min(len(r_fs), len(u_fs))):
        table.loc[(table[r_fs[i]] != table[u_fs[i]]), 'match_' + n_fs[i]] = -1
        table.loc[(table[r_fs[i]] == table[u_fs[i]]) | table[r_fs[i]], 'match_' + n_fs[i]] = 0
        table.loc[(table[r_fs[i]] == table[u_fs[i]]) & table[u_fs[i]], 'match_' + n_fs[i]] = 3
    table['payment_score'] = table.match_cash+table.match_visa+table.match_mc_ec+table.match_am_exp+table.match_debit
    if remove:
        for i in range(min(len(r_fs), len(u_fs))):
            table = table.drop(['match_' + n_fs[i]], axis=1)
            table = table.drop([r_fs[i]], axis=1)
            table = table.drop([u_fs[i]], axis=1)
    return table

###########################
# FEATURE: PAIR PROXIMITY #
###########################

def feature_pair_proximity(table, remove=True, keep_distance=True):
    table['D_lat'] = abs(table.R_latitude - table.U_latitude)
    table['D_long'] = abs(table.R_longitude - table.U_longitude)
    table['distance'] = (((table.D_lat * (111132.954 - 559.822 * cos(2.0 * table.R_latitude) +
                                          1.175 * cos(4.0 * table.R_latitude)))**2) +
                         ((table.D_long * ((3.14159265359/180) * 6367449 * cos(table.R_latitude)))**2)
                         ) ** 0.5

    # Creating banded feature "Proximity"
    table.loc[table.distance >= 5000, 'proximity'] = 0
    table.loc[table.distance < 5500, 'proximity'] = 1
    table.loc[table.distance < 4000, 'proximity'] = 2
    table.loc[table.distance < 2900, 'proximity'] = 3
    table.loc[table.distance < 2200, 'proximity'] = 4
    table.loc[table.distance < 1700, 'proximity'] = 5
    table.loc[table.distance < 1200, 'proximity'] = 6
    table.loc[table.distance < 800, 'proximity'] = 7
    table.loc[table.distance < 500, 'proximity'] = 8
    table.loc[table.distance < 300, 'proximity'] = 9
    table.loc[table.distance < 150, 'proximity'] = 10
    table.loc[table.distance < 100, 'proximity'] = 11
    table.loc[table.distance < 70, 'proximity'] = 12
    table.loc[table.distance < 40, 'proximity'] = 13
    table.loc[table.distance < 25, 'proximity'] = 14

    # Cleaning up table
    if remove:
        for column in ['D_lat', 'D_long', 'R_latitude', 'R_longitude', 'U_latitude', 'U_longitude']:
            table = table.drop([column], axis=1)
    if not keep_distance:
        table = table.drop(['distance'], axis=1)
    return table

#########################
# FEATURE: AVAILABILITY #
#########################

def feature_availability(table, remove=True):
    table['avg_hours'] = (table.R_weekday_hrs + table.R_sat_hrs + table.R_sun_hrs) / 3
    table['days_open'] = 0
    table.loc[table.R_weekday_hrs > 0, 'days_open'] += 5
    table.loc[table.R_sat_hrs > 0, 'days_open'] += 1
    table.loc[table.R_sun_hrs > 0, 'days_open'] += 1

    # Cleaning up table
    if remove:
        for column in ['R_weekday_hrs', 'R_sat_hrs', 'R_sun_hrs']:
            table = table.drop([column], axis=1)
    return table

##########################
# FEATURE: COMPATIBILITY #
##########################

def feature_alcohol(table, remove=True):
    table['match_alcohol'] = table.R_alcohol == table.U_alcohol
    table.loc[table.U_alcohol > 0, 'alcohol_score'] = table.R_alcohol * table.U_alcohol
    table.loc[table.U_alcohol == 0, 'alcohol_score'] = (2 - table.R_alcohol) + 0.5
    if remove:
        for column in ['R_alcohol', 'U_alcohol']:
            table = table.drop([column], axis=1)
    return table

def feature_noise(table, remove=True):
    table['match_quiet'] = table.R_quiet == table.U_quiet
    if remove:
        for column in ['R_quiet', 'U_quiet']:
            table = table.drop([column], axis=1)
    return table

def feature_smoking(table, remove=True):
    table.loc[table.U_smoking, 'smoking_score'] = table.R_smoking
    table.loc[table.U_smoking == False, 'smoking_score'] = 3 - table.R_smoking
    table['U_smoker'] = table.U_smoking
    if remove:
        for column in ['R_smoking', 'U_smoking']:
            table = table.drop([column], axis=1)
    return table

def feature_dress(table, remove=True):
    table['match_dress'] = table.R_formal_dress == table.U_formal_dress
    if remove:
        for column in ['R_formal_dress', 'U_formal_dress']:
            table = table.drop([column], axis=1)
    return table

def feature_price(table, remove=True):
    table['price_score'] = 2 - abs(table.U_budget - table.R_price)
    if remove:
        for column in ['R_price', 'U_budget']:
            table = table.drop([column], axis=1)
    return table

def feature_car_parking(table, remove=True):
    table['parking_score'] = 0
    table.loc[table.R_park_free & (table.U_transport < 2), 'parking_score'] = 1
    table.loc[table.R_park_none & (table.U_transport < 2), 'parking_score'] = 2
    table.loc[table.R_park_free & (table.U_transport == 2), 'parking_score'] = 2
    table.loc[table.R_park_paid & (table.U_transport == 2), 'parking_score'] = 3
    if remove:
        for column in ['R_park_free', 'R_park_paid', 'R_park_none', 'U_transport']:
            table = table.drop([column], axis=1)
    return table

############################
# FEATURE: CUISINE OVERLAP #
############################

def feature_cuisines(table, remove=True):
    values = []
    for i in range(len(table)):
        x = (table.iloc[i]['R_cuisine']).replace(' ', '').split(',')
        y = (table.iloc[i]['U_cuisine']).replace(' ', '').split(',')
        score = 0
        for dish in x:
            if dish in y:
                score += 3
        score += len(y)/5
        values.append(score)
    table['cuisine_score'] = values
    if remove:
        for column in ['R_cuisine', 'U_cuisine']:
            table = table.drop([column], axis=1)
    return table

##############################
# FEATURE: AVERAGE SUBRATING #
##############################

def feature_subrating(table, remove=True):
    table['subrating'] = (table.food_rating + table.service_rating + 1) // 2
    if remove:
        for column in ['food_rating', 'service_rating']:
            table = table.drop([column], axis=1)
    return table
