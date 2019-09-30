#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Andrew Floyd, Daniel Fuchs
Course: CS3001 - Dr. Fu
Data Science Competition Project
"""
import pandas as pd

####################
# HELPER FUNCTIONS #
####################

def apply_rank(frame, category_name, categories, forced_rank=None):
    if not forced_rank:
        for i in range(len(categories)):
            frame.loc[frame[category_name] == categories[i], category_name] = i
    else:
        for i, rank in zip(range(len(categories)), forced_rank):
            frame.loc[frame[category_name] == categories[i], category_name] = rank
    return frame

#####################
# RESTAURANT TABLES #
#####################

def clean_loc_accepts():
    def payment_mapping(payment):  # Note; maps payment methods no user uses as "misc".
        payment_types = ['cash', 'VISA', 'MasterCard-Eurocard', 'American_Express', 'bank_debit_cards', 'checks',
                         'Discover', 'Carte_Blanche', 'Diners_Club', 'Visa', 'Japan_Credit_Bureau', 'gift_certificates']
        p_mapping = ['cash', 'visa', 'mc_ec', 'am_exp', 'debit', 'check',
                     'misc', 'misc', 'misc', 'visa', 'misc', 'misc']
        return p_mapping[payment_types.index(payment)]

    t = pd.read_csv("chefmozaccepts.csv")
    t['Rpayment'] = t['Rpayment'].map(payment_mapping)
    categories = ['cash', 'visa', 'mc_ec', 'am_exp', 'debit', 'check']
    for category in categories:
        c_name = 'accepts_' + category
        t.loc[t.Rpayment == category, c_name] = True
        t.loc[t.Rpayment != category, c_name] = False
    t = t.groupby('placeID').any()
    t = t.drop(['Rpayment'], axis=1)
    return t

def clean_loc_cuisine():
    t = pd.read_csv("chefmozcuisine.csv")
    t = t.groupby(['placeID']).aggregate(lambda x: str(list(x)).replace('[', '').replace(']', '').replace('\'', ''))
    return t

def clean_loc_hours():
    t = pd.read_csv("chefmozhours4.csv").drop_duplicates()
    # t['opens'] = t['hours'].map(lambda x: int(str(x)[:2])) Could be used to get opening time.
    t['hours'] = t['hours'].map(lambda x: (24 + int(x[6:8]) - int(x[:2])) % 24)
    t.loc[(t.hours > 0) & (t.days == 'Mon;Tue;Wed;Thu;Fri;'), 'weekdays'] = t.hours
    t.loc[t.weekdays.isnull(), 'weekdays'] = 0
    t.loc[(t.hours > 0) & (t.days == 'Sat;'), 'sat_hours'] = t.hours
    t.loc[t.sat_hours.isnull(), 'sat_hours'] = 0
    t.loc[(t.hours > 0) & (t.days == 'Sun;'), 'sun_hours'] = t.hours
    t.loc[t.sun_hours.isnull(), 'sun_hours'] = 0
    t = t[['placeID', 'weekdays', 'sat_hours', 'sun_hours']].groupby('placeID').max()
    return t

def clean_loc_parking():
    def parking_mapping(payment):  # Note; maps payment methods no "user" uses as "misc".
        parking_types = ['public', 'none', 'yes', 'valet parking', 'fee', 'street', 'validated parking']
        k_mapping = ['free', 'none', 'free', 'paid', 'paid', 'free', 'paid']
        return k_mapping[parking_types.index(payment)]

    t = pd.read_csv("chefmozparking.csv")
    t['parking_lot'] = t['parking_lot'].map(parking_mapping)
    categories = ['free', 'paid']
    for category in categories:
        c_name = category + '_parking'
        t.loc[t.parking_lot == category, c_name] = True
        t.loc[t.parking_lot != category, c_name] = False
    t = t.groupby('placeID').any()
    t = t.drop(['parking_lot'], axis=1)
    t['no_parking'] = (t.free_parking | t.paid_parking) == False
    return t

def clean_loc_geo():
    t = pd.read_csv("geoplaces2.csv")
    t = t[['placeID', 'latitude', 'longitude', 'alcohol', 'smoking_area', 'dress_code', 'accessibility', 'price',
           'Rambience', 'franchise', 'area', 'other_services']]
    apply_rank(t, 'smoking_area', ['none', 'only at bar', 'permitted', 'section', 'not permitted'], [0, 1, 3, 2, 0])
    apply_rank(t, 'accessibility', ['no_accessibility', 'partially', 'completely'])
    apply_rank(t, 'alcohol', ['No_Alcohol_Served', 'Wine-Beer', 'Full_Bar'])
    apply_rank(t, 'other_services', ['none', 'Internet', 'variety'])
    apply_rank(t, 'price', ['low', 'medium', 'high'])
    t['Rformal_dress'] = t.dress_code == 'formal'
    t = t.drop(['dress_code'], axis=1)
    t['Rquiet'] = t.Rambience == 'quiet'
    t = t.drop(['Rambience'], axis=1)
    t['open_area'] = t.area == 'open'
    t = t.drop(['area'], axis=1)
    t['Rlatitude'] = t.latitude
    t = t.drop(['latitude'], axis=1)
    t['Rlongitude'] = t.longitude
    t = t.drop(['longitude'], axis=1)
    t['franchise'] = t.franchise == 't'
    return t

#######################
# USER-RELATED TABLES #
#######################

def clean_user_cuisine():
    t = pd.read_csv("usercuisine.csv")
    t.columns = ['userID', 'cuisine']
    t = t.groupby(['userID']).aggregate(lambda x: str(list(x)).replace('[', '').replace(']', '').replace('\'', ''))
    return t

def clean_user_payment():
    def payment_mapping(payment):
        payment_types = ['cash', 'VISA', 'MasterCard-Eurocard', 'American_Express', 'bank_debit_cards']
        p_mapping = ['cash', 'visa', 'mc_ec', 'am_exp', 'debit']
        return p_mapping[payment_types.index(payment)]

    t = pd.read_csv("userpayment.csv")
    t['Upayment'] = t['Upayment'].map(payment_mapping)
    categories = ['cash', 'visa', 'mc_ec', 'am_exp', 'debit']
    for category in categories:
        c_name = 'uses_' + category
        t.loc[t.Upayment == category, c_name] = True
        t.loc[t.Upayment != category, c_name] = False
    t = t.groupby('userID').any()
    t = t.drop(['Upayment'], axis=1)
    return t

def clean_user_profile():
    t = pd.read_csv("userprofile.csv")
    t = t[['userID', 'latitude', 'longitude', 'smoker', 'drink_level', 'dress_preference', 'ambience', 'transport',
           'marital_status', 'birth_year', 'interest', 'personality', 'activity', 'weight', 'budget']]
    for column in ['smoker', 'dress_preference', 'ambience', 'transport', 'marital_status', 'activity', 'budget']:
        t[column].fillna(t[column].mode()[0], inplace=True)
        t.loc[t[column] == '?', column] = t[column].mode()[0]
    apply_rank(t, 'personality', ['thrifty-protector', 'hunter-ostentatious', 'hard-worker', 'conformist'])
    apply_rank(t, 'interest', ['none', 'retro', 'variety', 'eco-friendly', 'technology'])
    apply_rank(t, 'drink_level', ['abstemious', 'casual drinker', 'social drinker'])
    apply_rank(t, 'transport', ['on foot', 'public', 'car owner'])
    apply_rank(t, 'budget', ['low', 'medium', 'high'])
    apply_rank(t, 'activity', ['student', 'unemployed', 'working-class', 'professional'], [0, 0, 1, 2])
    apply_rank(t, 'dress_preference', ['informal', 'formal', 'no preference', 'elegant'], [0, 1, 0, 1])
    t['formal_dress'] = t.dress_preference == 1
    t = t.drop(['dress_preference'], axis=1)
    t['quiet'] = t.ambience == 'solitary'
    t = t.drop(['ambience'], axis=1)
    t['married'] = t.marital_status == 'married'
    t = t.drop(['marital_status'], axis=1)
    t['age'] = 2018 - t.birth_year
    t = t.drop(['birth_year'], axis=1)
    t['smoker'] = t.smoker == 'true'
    return t

######################
# RATING INFORMATION #
######################

def clean_rating_table():
    t = pd.read_csv('rating_final.csv')
    t.columns = ['userID', 'placeID', 'rating', 'food_rating', 'service_rating']
    t['revID'] = t.index + 1
    return t
