#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Authors: Andrew Floyd, Daniel Fuchs
Course: CS3001 - Dr. Fu
Data Science Competition Project
"""
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, mean_squared_error, mean_absolute_error
from sklearn.neighbors import KNeighborsClassifier
from sklearn.linear_model import Lasso, LogisticRegression, LinearRegression, Ridge
from sklearn.cluster import KMeans
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from math import sqrt

from data_synthesis import synthesize_training_megatable, synthesize_testing_megatable, generate_testing_true_values
from data_preprocessing import *


def main():
    # Configuration Settings
    pd.set_option('display.width', 1000)
    pd.set_option('display.max_columns', 50)
    pd.set_option('display.max_rows', 200)

    # Generate Primary Feature Tables
    feature_tables = [synthesize_training_megatable(), synthesize_testing_megatable()]
    # analysis()

    # Apply Preprocessing
    processed_tables = []
    for feature_table in feature_tables:
        table = mass_feature_rename(feature_table)
        table = feature_payment_score(table, remove=True)
        table = feature_pair_proximity(table, remove=True, keep_distance=False)
        table = feature_availability(table, remove=True)
        table = feature_smoking(table, remove=True)
        table = feature_alcohol(table, remove=True)
        table = feature_noise(table, remove=True)
        table = feature_dress(table, remove=True)
        table = feature_price(table, remove=True)
        table = feature_car_parking(table, remove=True)
        table = feature_cuisines(table, remove=True)
        table = feature_subrating(table, remove=False)
        table = quantify_features(table)
        table = finalize_feature_selections(table)
        processed_tables.append(table)
    training_data = processed_tables[0]
    testing_data = processed_tables[1]

    # Create Model
    m_name = 'logistic Regression'.upper()
    if m_name == 'GAUSS':
        model = GaussianNB()
    elif m_name == 'DECISION TREE':
        model = DecisionTreeClassifier()
    elif m_name == 'LASSO':
        model = Lasso(alpha=0.01)
    elif m_name == 'RIDGE':
        model = Ridge(alpha=0.01)
    elif m_name == 'KNN':
        model = KNeighborsClassifier(n_neighbors=7, algorithm='auto', weights='distance')
    else:
        model = LogisticRegression(solver='newton-cg', multi_class='multinomial', C=1.5)

    # Train / Apply Model
    model.fit(extract_features(training_data), training_data['RATING'])
    predictions = np.rint(model.predict(extract_features(testing_data)))

    # Evaluate Performance
    true_values = mass_feature_rename(generate_testing_true_values())['RATING']
    print("Execution complete. The performance of %s is as follows:" % m_name)
    print("Accuracy:", accuracy_score(true_values, predictions))
    print("RMSE:", sqrt(mean_squared_error(true_values, predictions)))
    print("MAE:", mean_absolute_error(true_values, predictions))

    # Show Results
    show_results = False
    if show_results:
        compare = pd.DataFrame()
        compare['exact'] = model.predict(extract_features(testing_data))
        compare['true'] = true_values
        compare['pred'] = predictions
        compare['diff'] = compare.true - compare.pred
        compare = compare.join(testing_data)
        print(compare)

    # Generate Figures
    # create_scatter(predictions, true_values, "LassoRegression")
    # create_scatter(predictions, true_values, "KNN")
    # create_results_bar()
    return accuracy_score(true_values, predictions)


def analysis():
    accepts = pd.read_csv("chefmozaccepts.csv")
    cuisine = pd.read_csv("chefmozcuisine.csv")
    hours4 = pd.read_csv("chefmozhours4.csv")
    parking = pd.read_csv("chefmozparking.csv")
    usercuisine = pd.read_csv("usercuisine.csv")
    payment = pd.read_csv("userpayment.csv")
    userprofile = pd.read_csv("userprofile.csv")
    geoplaces2 = pd.read_csv("geoplaces2.csv")

    R = pd.read_csv("rating_final.csv")

    # train = pd.read_csv("train.txt")
    # test = pd.read_csv("test.txt")

    ###########################################
    # Lets do some preprocessing for our tables#
    ##########################################

    # Accepts
    # Here is a graph showing the volume of payments accepted
    # print(accepts.info())
    T1PaymentPlot = accepts.Rpayment.value_counts().plot.bar()
    T1PaymentPlot.set_title('Volume of Payments Accepted')
    T1PaymentPlot.set_xlabel('Form of Payment')
    T1PaymentPlot.set_ylabel('Number of Restaurants using Payment')

    # Here we are creating a new table with each restaurant having there own row
    # This will tell us which restaurants accept different payments
    Accepts_Transform = pd.get_dummies(accepts, columns=['Rpayment'])
    # This command combines rows with the same placeID into one row per ID
    Accepts_Transform = Accepts_Transform.groupby('placeID', as_index=False).sum()

    # Cuisine
    # A look at how many unique Restaurant Types we have (Stats)
    print("Number of unique Restaurant:", len(cuisine.placeID.unique()))
    print("Number of unique Restaurant Types:", len(cuisine.Rcuisine.unique()))
    print("A List of all Restaurant Types:")
    print(cuisine.Rcuisine.unique())
    print()
    print()
    # Bar Chart of top 20 Restaurant Types Volume
    T2CuisinePlot = cuisine.Rcuisine.value_counts()[:20].plot.bar()
    T2CuisinePlot.set_title('Volume of Top 20 Restaurant Types')
    T2CuisinePlot.set_xlabel('Restaurant Type', size=15)
    T2CuisinePlot.set_ylabel('Number of Restaurants')
    # Dummy chart transformation
    Cuisine_Transform = pd.get_dummies(cuisine, columns=['Rcuisine'])
    Cuisine_Transform = Cuisine_Transform.groupby('placeID', as_index=False).sum()

    # Hours4
    # Stats
    print("Number of unique Restaurants in Hours Table:", len(hours4.placeID.unique()))
    print("Number of unique times:", len(hours4.hours.unique()))
    print("List of Unique days:")
    print(hours4.days.unique())
    print()
    print()
    # Not sure if this data will be of much use

    # Parking
    # Stats
    print("Number of unique Restaurants in Parking Table:", len(parking.placeID.unique()))
    print()
    T4ParkingPlot = parking.parking_lot.value_counts().plot.bar()
    T4ParkingPlot.set_title('Volume of Parking Types', size=14)
    T4ParkingPlot.set_xlabel('Parking Type', size=12)
    T4ParkingPlot.set_ylabel('Count', size=12)
    # Dummy Transformation
    Parking_Transform = pd.get_dummies(parking, columns=['parking_lot'])
    Parking_Transform = Parking_Transform.groupby('placeID', as_index=False).sum()

    # UserCuisine
    # Stats
    print("Number of unique users in usercuisine table:", len(usercuisine.userID.unique()))
    print("Number of unique user restuarant categories:", len(usercuisine.Rcuisine.unique()))
    print("List of unique user restuarant categories:")
    print(usercuisine.Rcuisine.unique())
    print()
    print()
    # Bar Chart of top 20 most popular types
    T5UserCuisinePlot = usercuisine.Rcuisine.value_counts()[:20].plot.bar()
    T5UserCuisinePlot.set_title('Volume of Top 20 User Favorite Restaurant Types', size=14)
    T5UserCuisinePlot.set_xlabel('Restaurant Type', size=12)
    T5UserCuisinePlot.set_ylabel('Count', size=12)
    # Dummy Transformation
    UserCuisine_Transform = pd.get_dummies(usercuisine, columns=['Rcuisine'])
    UserCuisine_Transform = UserCuisine_Transform.groupby('userID', as_index=False).sum()

    # User Payments
    # Stats
    print("Number of unique users in payment table:", len(payment.userID.unique()))
    print("Number of unique user payment categories:", len(payment.Upayment.unique()))
    print("List of unique user payment categories:")
    print(payment.Upayment.unique())
    print()
    print()
    # Bar Chart
    T6UserPaymentPlot = payment.Upayment.value_counts().plot.bar()
    T6UserPaymentPlot.set_title('Volume of User Payments per Type', size=14)
    T6UserPaymentPlot.set_xlabel('User Payment Types', size=12)
    T6UserPaymentPlot.set_ylabel('Count', size=12)
    # Dummy Transformation
    Payment_Transformation = pd.get_dummies(payment, columns=['Upayment'])
    Payment_Transformation = Payment_Transformation.groupby('userID', as_index=False).sum()

    # User Profile
    # First, we know that some of the values in this table are unknown, so we are replacing
    # '?' with nan
    userprofile = userprofile.replace('?', np.nan)
    # Stats
    print("Number of unique users in userprofile table:", len(userprofile.userID.unique()))
    print()
    # Trying to find how many are missing per column
    Nan_Count = userprofile.isnull().sum()
    missing_DF = pd.DataFrame({'columns': userprofile.columns, 'nullCount': Nan_Count})
    print(missing_DF)
    print()
    # print(missing_DF.info())
    Missing_Profile_Plot = missing_DF.plot.bar(x='columns', y='nullCount', legend=False)
    Missing_Profile_Plot.set_title('Amount of missing values in UserProfile Table per Attribute')
    Missing_Profile_Plot.set_xlabel('Attributes')
    Missing_Profile_Plot.set_ylabel('Number of ? values')
    # Replacing Nan values with the mode of the column
    for column in userprofile.columns:
        userprofile[column].fillna(userprofile[column].mode()[0], inplace=True)

    # print(userprofile.smoker.value_counts())

    # Below are pie chart visualizations of some of the user profile data
    '''pie_smoker = userprofile['smoker'].value_counts().plot(kind='pie', figsize=(5,5))
    pie_smoker.set_title('Smoker Distribution')
    pie_smoker.set_xlabel('')
    pie_smoker.set_ylabel('')

    pie_drink_level = userprofile['drink_level'].value_counts().plot(kind='pie', figsize=(5,5))
    pie_drink_level.set_title('Drinking Level Distribution')
    pie_drink_level.set_xlabel('')
    pie_drink_level.set_ylabel('')

    pie_dress_preference = userprofile['dress_preference'].value_counts().plot(kind='pie', figsize=(5,5))
    pie_dress_preference.set_title('Dress Preference Distribution')
    pie_dress_preference.set_xlabel('')
    pie_dress_preference.set_ylabel('')

    pie_marital_status = userprofile['marital_status'].value_counts().plot(kind='pie', figsize=(5,5))
    pie_marital_status.set_title('Marital Status Distribution')
    pie_marital_status.set_xlabel('')
    pie_marital_status.set_ylabel('')

    pie_interest = userprofile['interest'].value_counts().plot(kind='pie', figsize=(5,5))
    pie_interest.set_title('Interest Distribution')
    pie_interest.set_xlabel('')
    pie_interest.set_ylabel('')

    pie_budget = userprofile['budget'].value_counts().plot(kind='pie', figsize=(5,5))
    pie_budget.set_title('Budget Distribution')
    pie_budget.set_xlabel('')
    pie_budget.set_ylabel('')'''

    # Bar plot to look at personal info by birthyear (interest, personality, religion, activity)
    ProfileBirthBar = userprofile.groupby('birth_year')[
        'interest', 'personality', 'religion', 'activity'].nunique().plot.bar(figsize=(15, 5))
    ProfileBirthBar.set_title('Users Personal Info Based on Birthyear')
    ProfileBirthBar.set_xlabel('Birth Year')

    # Transforming profile data into integer values
    ProfileInt = userprofile.select_dtypes(include=['object'])
    encoder = LabelEncoder()
    ProfileInt = ProfileInt.apply(encoder.fit_transform, axis=0)
    ProfileInt = ProfileInt.drop(['userID'], axis=1)
    ProfileInt[['userID', 'latitude', 'longitude', 'birth_year', 'weight', 'height']] = userprofile[
        ['userID', 'latitude', 'longitude', 'birth_year', 'weight', 'height']]
    print(ProfileInt.head())
    print()

    # Geoplaces2
    # Stats
    print("Number of unique restaurants in geoplaces2:", len(geoplaces2.placeID.unique()))
    print()
    # Splitting geo data into to relevant features
    geo_relevant = geoplaces2[['placeID', 'alcohol', 'smoking_area', 'other_services', 'price']]
    print(geo_relevant.alcohol.value_counts())
    print()
    print(geo_relevant.smoking_area.value_counts())
    print()
    print(geo_relevant.other_services.value_counts())
    print()
    print(geo_relevant.price.value_counts())
    print()
    # Replacing ? values with Nan
    geoplaces2 = geoplaces2.replace('?', np.nan)
    Missing_Geo = geoplaces2.isnull().sum()
    missingGeo_DF = pd.DataFrame({'columns': geoplaces2.columns, 'nullCount': Missing_Geo})
    print(missingGeo_DF)
    print()
    # print(missingGeo_DF.info())
    Missing_Geo_Plot = missingGeo_DF.plot.bar(x='columns', y='nullCount', legend=False)
    Missing_Geo_Plot.set_title('Amount of missing values in Geoplaces2 Table per Attribute')
    Missing_Geo_Plot.set_xlabel('Attributes')
    Missing_Geo_Plot.set_ylabel('Number of ? values')
    # We are going to drop fax, zip and url due to the high level of Nan's
    geoplaces2_clean = geoplaces2.drop(['fax', 'zip', 'url'], axis=1)
    # Replacing Nan values with the mode of the column
    for column in geoplaces2_clean.columns:
        geoplaces2_clean[column].fillna(geoplaces2_clean[column].mode()[0], inplace=True)
    # Cleaning city names
    print(geoplaces2_clean.city.value_counts())
    print()
    geoplaces2_clean['city'] = geoplaces2_clean['city'].replace(
        ['San Luis Potosi', 'san luis potosi', 'san luis potosi ', 'san luis potos', 's.l.p', 'slp', 's.l.p.'],
        'san luis potosi')
    geoplaces2_clean['city'] = geoplaces2_clean['city'].replace(['Cuernavaca', 'cuernavaca'], 'cuernavaca')
    geoplaces2_clean['city'] = geoplaces2_clean['city'].replace(
        ['victoria', 'Ciudad Victoria', 'victoria ', 'Cd. Victoria', 'Cd Victoria'], 'ciudad victoria')
    geoplaces2_clean['city'] = geoplaces2_clean['city'].replace(['Jiutepec'], 'jiutepec')
    geoplaces2_clean['city'] = geoplaces2_clean['city'].replace(['Soledad'], 'soledad')
    print(geoplaces2_clean.city.value_counts())
    print()
    # Cleaning state names
    print(geoplaces2_clean.state.value_counts())
    print()
    geoplaces2_clean['state'] = geoplaces2_clean['state'].replace(
        ['SLP', 'San Luis Potosi', 'san luis potosi', 'slp', 'S.L.P.', 'san luis potos', 's.l.p.'], 'san luis potosi')
    geoplaces2_clean['state'] = geoplaces2_clean['state'].replace(['Morelos', 'morelos'], 'morelos')
    geoplaces2_clean['state'] = geoplaces2_clean['state'].replace(['tamaulipas', 'Tamaulipas'], 'tamaulipas')
    print(geoplaces2_clean.state.value_counts())
    print()
    # Cleaning country names
    print(geoplaces2_clean.country.value_counts())
    print()
    geoplaces2_clean['country'] = geoplaces2_clean['country'].replace(['Mexico', 'mexico'], 'mexico')
    print(geoplaces2_clean.country.value_counts())
    print()
    # Dummy Transformation (using encoder)
    GeoInt = geoplaces2_clean.select_dtypes(include=['object'])
    GeoInt = GeoInt.apply(encoder.fit_transform, axis=0)
    GeoInt[['placeID', 'latitude', 'longitude']] = geoplaces2_clean[['placeID', 'latitude', 'longitude']]
    pd.set_option('display.max_columns', None)
    print(GeoInt.head())
    
def create_scatter(predictions, true_values, model):
    df_test = pd.DataFrame(predictions, columns=['prediction'])
    plt.scatter(df_test.index, df_test.prediction, color='r', s=2)
    plt.scatter(true_values.index, true_values.RATING, color='b', s=2)
    plt.legend(loc=1)
    title = " Predicted Ratings v. Actual Ratings Scatter Plot"
    str_title = model + title
    plt.title(str_title)
    plt.xlabel('Index Value')
    plt.ylabel('Rating Value')
    savefig = model + '_scatter.png'
    plt.savefig(savefig, dpi=750)
    
def create_results_bar():
    ind = np.arange(6)
    results_acc = [0.650429799, 0.80515759, 0.66475644, 0.69340974, 0.773638968, 0.7707736389]
    results_rmse = [0.6855027644, 0.4697130746, 0.6149987769, 0.63336098, 0.48472365, 0.487670329]
    results_mse = [0.389684813, 0.203438295, 0.34957020057, 0.3381088825, 0.229226361, 0.2320916905]
    width = 0.15
    fig, ax = plt.subplots()
    results_bar1 = ax.bar(ind-width, results_acc, width, color='r')
    results_bar2 = ax.bar(ind, results_mse, width, color='b')
    results_bar3 = ax.bar(ind+ width, results_rmse, width, color='g')
    ax.set_xticks(ind)
    ax.set_xticklabels(('KNN', 'LogReg', 'GaussNB', 'DecTree', 'LassoReg', 'RidgeReg'))
    ax.set_xlabel('Model Type')
    ax.set_ylabel('Value')
    ax.legend((results_bar1[0], results_bar2[0], results_bar3[0]), ('Accuracy', 'MSE', 'RMSE'))
    ax.set_title('Accuracy, MSE and RMSE Results by Model')
    fig.savefig('results_bar.png', dpi=750)


# # # # # # # # # # # #
# # # END PROGRAM # # #
# # # # # # # # # # # #


if __name__ == '__main__':
    main()
