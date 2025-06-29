#!/usr/bin/env python
# coding: utf-8

# In[50]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import re
import os        
import base64
import logging
import copy

import shap
from scipy import stats


import sklearn # train the linear model
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, PowerTransformer, QuantileTransformer
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error, roc_auc_score, confusion_matrix, accuracy_score, f1_score, \
precision_score, recall_score, classification_report, roc_curve
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor, GradientBoostingRegressor, BaggingRegressor
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor, ExtraTreeRegressor
from sklearn.linear_model import LinearRegression, SGDRegressor, LogisticRegression
from sklearn.neighbors import KNeighborsRegressor
from sklearn import preprocessing
from sklearn.pipeline import make_pipeline, Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.model_selection import KFold
from sklearn.cluster import KMeans
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
#from sklearn.datasets import make_regression
from yellowbrick.cluster import KElbowVisualizer

# permutation feature importance with knn for regression
from sklearn.neighbors import KNeighborsRegressor
from sklearn.inspection import permutation_importance

from io import BytesIO
# from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator
# from descriptastorus.descriptors import rdNormalizedDescriptors, rdDescriptors
from plotly.graph_objects import Figure
import plotly.express as px
import plotly.io as pio
# from dash import Dash, Input, Output, dcc, html, no_update
# from jupyter_dash import JupyterDash

import missingno as msno
import umap
import umap.plot  # pip install umap-learn[plot]

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

# import warnings
# warnings.filterwarnings("ignore")
# generator = rdNormalizedDescriptors.RDKit2DNormalized()
# feature_list = []
# for name in generator.columns:
#     feature_list.append(name[0])

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


# In[51]:


# import gtda.diagrams as diag
# from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance
# from gtda.homology import VietorisRipsPersistence
# import gtda.graphs as gr
# from gtda.pipeline import Pipeline
# from gtda.plotting import plot_point_cloud, plot_heatmap, plot_diagram
# from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance
# from gtda.mapper import (
#     CubicalCover,
#     OneDimensionalCover,
#     make_mapper_pipeline,
#     Projection,
#     plot_static_mapper_graph,
#     plot_interactive_mapper_graph)
# from gtda.mapper import Eccentricity, Entropy

#importing rdkit modules
# from rdkit import rdBase, Chem
# from rdkit.Chem import AllChem, Draw, PandasTools, DataStructs
# from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
# from descriptastorus.descriptors import rdNormalizedDescriptors


#number of features for use in QRC. Takes the top nfeats values for valu
#nfeats = 18
nfeats = 8
subsamp_size = 200 #Size of subsamples we create for cross validation
n_clusters = 5
threshold_z = 3.5 #Z = 3.5 =  99.95%, Z=3 is 99.86% of all records
top_code_percent = .9995 #value to top code values too
shapsamp = 10#400 #Number of observations to include in SHAP, which takes a real
subnum =4
recs =100
actfile=4


# In[57]:


# df= pd.read_csv('/home/shared/data/merck/TrainingSet/ACT'+str(actfile)+'_competition_training.csv')


# # In[58]:


# df.Act.info(verbose=True, show_counts=True)


# # # EDA of Target Var and Make plot of Target Var
# # 
# # Skew and Kurtosis are both below the threshold for issues with this being interpreted as a normal distirubtion. 

# # In[59]:


# print(df['Act'].describe())

# print('Skew', df.Act.skew(), 'Kurtosis', df.Act.kurtosis())


# In[60]:


def removal_box_plot(df, column, threshold):
    sns.boxplot(df[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
 
    removed_outliers = df[df[column] <= threshold]
 
    sns.boxplot(removed_outliers[column])
    plt.title(f'Box Plot without Outliers of {column}')
    plt.show()
    return removed_outliers
 
 
threshold_value = 7
 
#no_outliers = removal_box_plot(df, 'Act', threshold_value)


# In[61]:


# z = np.abs(stats.zscore(df['Act']))

# outlier_indices = np.where(z > threshold_z)[0]
# outliers = df[z > threshold_z]
# print("Num records total",len(df), "Num Outliers", len(outliers), "% Outlier", (len(outliers)/len(df)*100), "%")
# top_code_percent #99.95%
# top_code_value = df["Act"].quantile(top_code_percent)
# print("Values over this value are top-coded to this value",top_code_value)

#Outliers don't look that bad, only 21, but top coding since sampling provides wild values and there was a value below zero
#df_test = df_feature.copy()
#df['Act'] = np.where(df['Act'] > top_code_value, top_code_value, df['Act']) #Top Code high outliers
#df['Act'] = np.where(df['Act'] < 0, 0, df['Act']) #remove values for TOX under 0, impossible


# # Standardize Data

# In[62]:


# scaler = preprocessing.StandardScaler()
# df_scaled = df.drop(['MOLECULE', 'Act'], axis=1)
# x_red = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index) 

# X_train_all, X_test_all, y_train_all, y_test_all = train_test_split \
# (x_red, df['Act'], test_size=0.2)#, random_state=2212) #.drop(['Act'])

# X_train_all.columns

kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))


def train_gaussian_process_with_kernel_matrix(X_train_all, X_test_all, y_train_all, y_test_all):
    # Define the RBF kernel
    kernel = RBF()

    # Initialize the Gaussian Process Regressor with the RBF kernel
    gpr = GaussianProcessRegressor(kernel=kernel) #, alpha=1e-3)

    # Fit the model to the training data
    gpr.fit(X_train_all, y_train_all)

    # Predict on the test data
    y_pred = gpr.predict(X_test_all)

    # Output the kernel matrix
    K = gpr.kernel_(X_train_all)

    print(f"Kernel Matrix:\n{K}")

    return gpr, y_pred, K

def removal_box_plot(df, column, threshold):
    sns.boxplot(df[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
 
    removed_outliers = df[df[column] <= threshold]
 
    sns.boxplot(removed_outliers[column])
    plt.title(f'Box Plot without Outliers of {column}')
    plt.show()
    return removed_outliers
 
 
threshold_value = 7
 
#no_outliers = removal_box_plot(df, 'Act', threshold_value)


# In[67]:


#dims = features.shape[1]
def nnet(X_train, X_test, y_train, y_test):
    dims = X_train.shape[1]
    
    
    # Convert to 2D PyTorch tensors
    X_train = torch.tensor(X_train.to_numpy(), dtype=torch.float32)
    y_train = torch.tensor(y_train.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    X_test = torch.tensor(X_test.to_numpy(), dtype=torch.float32)
    y_test = torch.tensor(y_test.to_numpy(), dtype=torch.float32).reshape(-1, 1)
    
    # Define the model
    model = nn.Sequential(
        nn.Linear(dims, 24),
        nn.ReLU(),
        nn.Linear(24, 12),
        nn.ReLU(),
        nn.Linear(12, 6),
        nn.ReLU(),
        nn.Linear(6, 1)
    )
     
    # loss function and optimizer
    loss_fn = nn.MSELoss()  # mean square error
    optimizer = optim.Adam(model.parameters(), lr=0.0001)
     
    n_epochs = 100   # number of epochs to run
    batch_size = 10  # size of each batch
    batch_start = torch.arange(0, len(X_train), batch_size)
     
    # Hold the best model
    best_mse = np.inf   # init to infinity
    best_weights = None
    history = []
     
    for epoch in range(n_epochs):
        model.train()
        with tqdm.tqdm(batch_start, unit="batch", mininterval=0, disable=True) as bar:
            bar.set_description(f"Epoch {epoch}")
            for start in bar:
                # take a batch
                X_batch = X_train[start:start+batch_size]
                y_batch = y_train[start:start+batch_size]
                # forward pass
                y_pred = model(X_batch)
                loss = loss_fn(y_pred, y_batch)
                # backward pass
                optimizer.zero_grad()
                loss.backward()
                # update weights
                optimizer.step()
                # print progress
                bar.set_postfix(mse=float(loss))
        # evaluate accuracy at end of each epoch
        model.eval()
        y_pred = model(X_test)
        mse = loss_fn(y_pred, y_test)
        mse = float(mse)
        history.append(mse)
        if mse < best_mse:
            best_mse = mse
            best_weights = copy.deepcopy(model.state_dict())
     
    # restore model and return best accuracy
    model.load_state_dict(best_weights)
    print("MSE: %.2f" % best_mse)
    print("RMSE: %.2f" % np.sqrt(best_mse))
    plt.plot(history)
    plt.show()
     
#    model.eval()
#    with torch.no_grad():
#        # Test out inference with 5 samples
#        for i in range(5):
#            X_sample = X_test_raw[i: i+1]
#            X_sample = scaler.transform(X_sample)
#            X_sample = torch.tensor(X_sample, dtype=torch.float32)
#            y_pred = model(X_sample)
#            print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")


# In[68]:


num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])


def prepare_model(algorithm, X_train, y_train, columns_to_keep): 
    model = Pipeline(steps=[('preprocessing', num_pipeline),('algorithm', algorithm)])
    model.fit(X_train.loc[:,columns_to_keep], y_train)
    return model

algorithms = [
                ['RandomForestRegressor',RandomForestRegressor \
                 (random_state=42, n_estimators = 300, max_depth = 30, criterion = 'absolute_error')], 
                ['AdaBoostRegressor',AdaBoostRegressor()], 
                ['GradientBoostingRegressor',GradientBoostingRegressor()], 
                ['BaggingRegressor',BaggingRegressor()], 
                ['SVR',SVR()], 
                ['DecisionTreeRegressor',DecisionTreeRegressor()], 
                ['ExtraTreeRegressor',ExtraTreeRegressor()], 
                ['LinearRegression',LinearRegression()], 
                ['SGDRegressor',SGDRegressor()], 
                ['KNeighborsRegressor', KNeighborsRegressor()],
                ['GaussianProcessRegressor', GaussianProcessRegressor(kernel=kernel, alpha=0.0)]
            ]

names = []
times = []
mses = []
maes = []
r2 = []

def runalgos2(X_train, X_test, y_train, y_test, columns_to_keep):
    names = []
    times = []
    mses = []
    maes = []
    r2 = []

    # Define kernel for GaussianProcessRegressor
    kernel = RBF()

    # Define pipelines for each algorithm
    pipelines = {
        'GaussianProcessRegressor': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('regressor', GaussianProcessRegressor(kernel=kernel, alpha=0.000001))
        ]),
        'LinearRegression': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('regressor', LinearRegression())
        ]),
        'RandomForestRegressor': Pipeline([
            ('imputer', SimpleImputer(strategy='mean')),
            ('regressor', RandomForestRegressor())
        ])
    }

    # Train and evaluate each pipeline
    for name, model in pipelines.items():
        start_time = time.time()
        model.fit(X_train[columns_to_keep], y_train)
        end_time = time.time()

        # Align the columns of the test set with the training set
        X_test_aligned = X_test.reindex(columns=columns_to_keep, fill_value=0)

        # Make predictions
        pred = model.predict(X_test_aligned)

        # Calculate metrics
        mse = mean_squared_error(y_test, pred)
        mae = mean_absolute_error(y_test, pred)
        r2_val = r2_score(y_test, pred)

        # Append results
        names.append(name)
        times.append(end_time - start_time)
        mses.append(mse)
        maes.append(mae)
        r2.append(r2_val)

    # Create results dictionary
    results_dict = {'Algorithm': names, 'MSE': mses, 'MAE': maes, 'R2': r2, 'Time': times}
    print(results_dict)

    return results_dict

def runalgos1(X_train, X_test, y_train, y_test, columns_to_keep):
    # from sklearn.pipeline import Pipeline
    # from sklearn.impute import SimpleImputer
    # from sklearn.ensemble import RandomForestRegressor
    # from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
    # import time

    names = []
    times = []
    mses = []
    maes = []
    r2 = []

    # Define the model pipeline
    model = Pipeline([
        ('imputer', SimpleImputer(strategy='mean')),
        ('regressor', RandomForestRegressor())
    ])

    # Fit the model
    start_time = time.time()
    model.fit(X_train[columns_to_keep], y_train)
    end_time = time.time()

    # Align the columns of the test set with the training set
    X_test_aligned = X_test.reindex(columns=columns_to_keep, fill_value=0)

    # Make predictions
    pred = model.predict(X_test_aligned)

    # Calculate metrics
    mse = mean_squared_error(y_test, pred)
    mae = mean_absolute_error(y_test, pred)
    r2_val = r2_score(y_test, pred)

    # Append results
    names.append('RandomForestRegressor')
    times.append(end_time - start_time)
    mses.append(mse)
    maes.append(mae)
    r2.append(r2_val)

    results_dict = {'Algorithm': names, 'MSE': mses, 'MAE': maes, 'R2': r2, 'Time': times}
    print(results_dict)

    return results_dict


def bestmod(modname, X_train, X_test, y_train, y_test):
    best_model=modname()
    
    best_model.fit(X_train, y_train)
    print(best_model.score(X_train, y_train))
    print(best_model.score(X_test, y_test))
    
    best_model_pred = best_model.predict(X_test)
    best_model_mse = mean_squared_error(y_test, best_model_pred)
    print(best_model_mse)
    
    y_pred = best_model.predict(X_test)
    
    r2=r2_score(y_test, best_model_pred)
    print("R2:",r2)
    
    sns.regplot(x = best_model_pred, y = y_test)
    return best_model.score(X_train, y_train), best_model.score(X_test, y_test),  best_model_mse, r2


# In[69]:


def shapgrad(model, X_train, y_train, nfeats=10, actfile='default', recs='default'):
    rdr = model()  # Initialize the model
    rdr.fit(X_train, y_train)  # Fit the model
    explainer = shap.TreeExplainer(rdr, data=X_train)  # Create SHAP explainer
    shap_values = explainer.shap_values(X_train, check_additivity=False)  # Get SHAP values
    
    # Visualize the model's dependence on the first feature
    shap.summary_plot(shap_values, X_train)
    feature_names = X_train.columns
    
    rf_resultX = pd.DataFrame(shap_values, columns=feature_names)  # Create DataFrame of SHAP values
    vals = np.abs(rf_resultX.values).mean(0)  # Calculate mean absolute SHAP values
    
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                   columns=['col_name', 'feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'], ascending=False, inplace=True)
    columns_to_keep = shap_importance.iloc[0:nfeats, 0].values.tolist()
    kfoldin = X_train.loc[:, columns_to_keep]
    
    # Step 1: Check if the index has duplicates and reset the index if necessary
    if kfoldin.index.duplicated().any():
        kfoldin = kfoldin.reset_index(drop=True)
        kfoldin = kfoldin[~kfoldin.index.duplicated(keep='first')]
    
    kfoldin['Act'] = y_train.values  # Ensure y_train is aligned with kfoldin
    X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split(
        kfoldin.loc[:, columns_to_keep], y_train, test_size=0.2)  # Split the data
    
    shap.summary_plot(shap_values, X_train, plot_type='bar')
    plt.show()
    #plt.savefig(f'/Users/dabeaulieu/Documents/Initiatives/quantum/machine_learning/notebooks/quera/regression/merck/qrcscript/figures/actfile_{actfile}recs{recs}fig5_shap_grad.png')
    
    return X_train_shap, X_test_shap, y_train_shap, y_test_shap, columns_to_keep, kfoldin


# In[70]:


def shapkern(model, X_train, y_train):
    rdr = model() #GradientBoostingRegressor()
    rdr.fit(X_train, y_train)
    # explain the model's predictions using SHAP
    explainer = shap.KernelExplainer(rdr.predict, X_train)
    shap_values = explainer.shap_values(X_train) #Won't work with X_test, too small

    # visualize the model's dependence on the first feature
    shap.summary_plot(shap_values, X_train)
    feature_names = X_train.columns
    
    rf_resultX = pd.DataFrame(shap_values, columns = feature_names)
    vals = np.abs(rf_resultX.values).mean(0)
    
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    shap_importance.head(nfeats)
    columns_to_keep = shap_importance.iloc[0:nfeats, 0].values.tolist()
    kfoldin = X_train.loc[:, columns_to_keep]
    targ = y_train
    X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split \
    (kfoldin.loc[:, columns_to_keep], targ, test_size=0.2)#, random_state=2212)
    

    shap.summary_plot(shap_values, X_train, plot_type='bar')
    plt.show()
    #plt.savefig('/Users/dabeaulieu/Documents/Initiatives/quantum/machine_learning/notebooks/quera/regression/merck/qrcscript/figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig5_shap_grad.png')
    return X_train_shap, X_test_shap, y_train_shap, y_test_shap, columns_to_keep, kfoldin


# In[71]:



all_results_df = pd.DataFrame()

#acts = [14] #,14,15,9]
#subs =  [1,2,3,4,5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25] #[1,2,3,4,5]
#recs = [100] #, 200, 800] 

acts = [14] #,14,15,9]
subs = [1,2,]#3,4,5] #,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25]
recs = [100] #[800] 

for act in acts:
    for sub in subs:
        for rec in recs:
            print(f"Processing SHAP Classical act: {act}, sub: {sub}, rec: {rec}")
            version =5 #3
            # Read the data
            df_train = pd.read_csv(f'./X_train_{rec}rec_sub{sub}act{act}v{version}.csv')
            df_test = pd.read_csv(f'./X_test_{rec}rec_sub{sub}act{act}v{version}.csv')
            df_ytrn = pd.read_csv(f'./y_train_{rec}rec_sub{sub}act{act}v{version}.csv', usecols=['Act'])
            df_ytst = pd.read_csv(f'./y_test_{rec}rec_sub{sub}act{act}v{version}.csv', usecols=['Act'])

            # Run algorithms and store results
            columns_to_keep=df_train.columns
            results_dict_qrc = runalgos2(df_train, df_test, df_ytrn, df_ytst, columns_to_keep)
            results_df_qrc = pd.DataFrame(results_dict_qrc)
            results_df_qrc['act'] = act
            results_df_qrc['sub'] = sub
            results_df_qrc['rec'] = rec
            results_df_qrc['source'] = 'results_dict_qrc'
            all_results_df = pd.concat([all_results_df, results_df_qrc], ignore_index=True)
            version = 5 #version = 2
            # Import Data Embeddings back from Julia QRC
            train_emb = pd.read_csv(f"./records{rec}/merck_train_embeddings_recs{rec}rec_sub{sub}act{act}v{version}.csv").T
            test_emb = pd.read_csv(f"./records{rec}/merck_test_embeddings_recs{rec}rec_sub{sub}act{act}v{version}.csv").T
            y_train_emb = pd.read_csv(f"./records{rec}/merck_train_outcomes_recs{rec}rec_sub{sub}act{act}v{version}.csv", header=None)
            y_test_emb = pd.read_csv(f"./records{rec}/merck_test_outcomes_recs{rec}rec_sub{sub}act{act}v{version}.csv", header=None)
            print(f"Processing Embedding act: {act}, sub: {sub}, rec: {rec}")

            #X_train_shapemb, X_test_shapemb, y_train_shapemb, y_test_shapemb, columns_to_keep, kfoldin = shapgrad(RandomForestRegressor, train_emb, y_train_emb)
            X_train_shapemb, X_test_shapemb, y_train_shapemb, y_test_shapemb, columns_to_keep, kfoldin = shapgrad(RandomForestRegressor, train_emb, y_train_emb)
            columns_to_keep=X_train_shapemb.columns
            results_dict_qrcemb = runalgos2(X_train_shapemb, X_test_shapemb, y_train_shapemb, y_test_shapemb, columns_to_keep)
            results_df_qrcemb = pd.DataFrame(results_dict_qrcemb)
            results_df_qrcemb['act'] = act
            results_df_qrcemb['sub'] = sub
            results_df_qrcemb['rec'] = rec
            results_df_qrcemb['source'] = 'results_dict_qrcemb'
            all_results_df = pd.concat([all_results_df, results_df_qrcemb], ignore_index=True)
            version = 5 #3
            print(f"Processing Linear Embedding act: {act}, sub: {sub}, rec: {rec}")

            # Import Data Embeddings back from Julia QRC (Linear)
            train_emb_lin = pd.read_csv(f"./records{rec}/merck_train_emb_lin_rec{rec}rec_sub{sub}act{act}v{version}.csv", header=None).T
            test_emb_lin = pd.read_csv(f"./records{rec}/merck_test_emb_lin_rec{rec}rec_sub{sub}act{act}v{version}.csv", header=None).T
            y_train_emb_lin = pd.read_csv(f"./records{rec}/merck_train_outcomes_lin_rec{rec}rec_sub{sub}act{act}v{version}.csv", header=None)
            y_test_emb_lin = pd.read_csv(f"./records{rec}/merck_test_outcomes_lin_rec{rec}rec_sub{sub}act{act}v{version}.csv", header=None)

            #X_train_shapemblin, X_test_shapemblin, y_train_shapemblin, y_test_shapemblin, columns_to_keep_emblin, kfoldin_emblin = shapgrad(RandomForestRegressor, train_emb_lin, y_train_emb_lin)
            X_train_shapemblin, X_test_shapemblin, y_train_shapemblin, y_test_shapemblin, columns_to_keep_emblin, kfoldin_emblin = shapgrad(RandomForestRegressor, train_emb_lin, y_train_emb_lin)
            columns_to_keep=X_train_shapemblin.columns
            results_dict_qrcemblin = runalgos2(X_train_shapemblin, X_test_shapemblin, y_train_shapemblin, y_test_shapemblin, columns_to_keep)
            results_df_qrcemblin = pd.DataFrame(results_dict_qrcemblin)
            results_df_qrcemblin['act'] = act
            results_df_qrcemblin['sub'] = sub
            results_df_qrcemblin['rec'] = rec
            results_df_qrcemblin['source'] = 'results_dict_qrcemblin'
            all_results_df = pd.concat([all_results_df, results_df_qrcemblin], ignore_index=True)

# Output the master DataFrame to a CSV file
all_results_df.to_csv('./allalgos_2025_04_25_5subsamps_act14_800_recs_v3.csv', index=False)

