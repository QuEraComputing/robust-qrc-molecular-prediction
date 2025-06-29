#!/usr/bin/env python
# coding: utf-8

# In[1]:


#number of fseatures for use in QRC. Takes the top nfeats values for valu
nfeats = 18
#subsamp_size = 100 #Size of subsamples we create for cross validation
n_clusters = 5
threshold_z = 3.5 #Z = 3.5 =  99.95%, Z=3 is 99.86% of all records
top_code_percent = .9995 #value to top code values  X_train_sub1, X_test_sub1, y_train_sub1, y_test_sub1too
shapsamp = 100 #200 #Number of observations to include in SHAP, which takes a real

subnum = -100 #1
recs = 200 #100
actfile=4 #14
version = 3 #5

data_dir = './DATA/' #Directory to save generated data files


print("ACT File: ", actfile)
print("Begin Data Prep Process Merck Activity Challenge QRC Prep")
# In[2]:


import numpy as np # linear algebra
import pandas as pd # data processing, CSV file I/O (e.g. pd.read_csv)
import time
import re
import os        
import base64
import logging
import copy
import gc  # Add garbage collector for memory management

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
#from descriptastorus.descriptors.DescriptorGenerator import MakeGenerator#
#from descriptastorus.descriptors import rdNormalizedDescriptors, rdDescriptors

from plotly.graph_objects import Figure
import plotly.express as px
#from dash import Dash, Input, Output, dcc, html, no_update
#from jupyter_dash import JupyterDash

import missingno as msno
import umap
import umap.plot  # pip install umap-learn[plot]

from matplotlib import pyplot
import matplotlib.pyplot as plt
import seaborn as sns
#%matplotlib inline

import warnings
warnings.filterwarnings("ignore")

# Add memory monitoring
import psutil
def print_memory_usage():
    process = psutil.Process(os.getpid())
    memory_info = process.memory_info()
    memory_mb = memory_info.rss / 1024 / 1024
    print(f"Current memory usage: {memory_mb:.2f} MB")
    return memory_mb
# generator = rdNormalizedDescriptors.RDKit2DNormalized()
# feature_list = []
# for name in generator.columns:
#     feature_list.append(name[0])

import torch
import torch.nn as nn
import torch.optim as optim
import tqdm


# In[3]:


# import gtda.diagrams as diag
# from gtda.diagrams import Scaler, Filtering, PersistenceEntropy, BettiCurve, PairwiseDistance
# from gtda.homology import VietorisRipsPersistence
# import gtda.graphs as gr
# from gtda.pipeline import Pipeline
# from gtda.plotting import plot_point_cloud, plot_heatmap
# from gtda.graphs import KNeighborsGraph, GraphGeodesicDistance
# from gtda.mapper import (
#     CubicalCover,
#     OneDimensionalCover,
#     make_mapper_pipeline,
#     Projection,
#     plot_static_mapper_graph,
#     plot_interactive_mapper_graph)
# from gtda.mapper import Eccentricity, Entropy


# In[4]:


#importing rdkit modules
# from rdkit import rdBase, Chem
# from rdkit.Chem import AllChem, Draw, PandasTools, DataStructs
# from rdkit.Chem.Draw import IPythonConsole, rdMolDraw2D
# from descriptastorus.descriptors import rdNormalizedDescriptors


df_full = pd.read_csv(data_dir + 'TrainingSet/ACT'+str(actfile)+'_competition_training.csv')
df =  df_full.sample(n=1800)
print(df.shape)

# Clean up full dataset from memory
del df_full
gc.collect()

df.Act.info(verbose=True, show_counts=True)


# # EDA of Target Var and Make plot of Target Var
# Skew and Kurtosis are both below the threshold for issues with this being interpreted as a normal distirubtion. 

print(df['Act'].describe())
print('Skew', df.Act.skew(), 'Kurtosis', df.Act.kurtosis())

def removal_box_plot(df, column, threshold):
    sns.boxplot(df[column])
    plt.title(f'Original Box Plot of {column}')
    plt.show()
 
    removed_outliers = df[df[column] <= threshold]
 
    sns.boxplot(removed_outliers[column])
    plt.title(f'Box Plot without Outliers of {column}')
    #plt.show()
    plt.savefig('figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig2_boxplot_no_outliers_Acts_vs_D_58.png')

    return removed_outliers
 
 
threshold_value = 7
 
no_outliers = removal_box_plot(df, 'Act', threshold_value)

print('Histogram of Molecular Activity')
# add a 'best fit' line

plt.hist(df['Act'], bins=25)
plt.title(r'Histogram of Act')
plt.ylabel('Frequency')
plt.xlabel('Molecular Activty')

fig, ax = plt.subplots(figsize=(6, 4))
ax.scatter(df['Act'], df['D_58'])
ax.set_xlabel('Lethal Dose 50%')
ax.set_ylabel('D_58')
#plt.show()
plt.savefig('figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig1_scatter_Acts_vs_D_58.png')


z = np.abs(stats.zscore(df['Act']))

outlier_indices = np.where(z > threshold_z)[0]
outliers = df[z > threshold_z]
print("Num records total",len(df), "Num Outliers", len(outliers), "% Outlier", (len(outliers)/len(df)*100), "%")
top_code_percent #99.95%
top_code_value = df["Act"].quantile(top_code_percent)
print("Values over this value are top-coded to this value",top_code_value)

#Outliers don't look that bad, only 21, but top coding since sampling provides wild values and there was a value below zero
#df_test = df_feature.copy()
#df['Act'] = np.where(df['Act'] > top_code_value, top_code_value, df['Act']) #Top Code high outliers
#df['Act'] = np.where(df['Act'] < 0, 0, df['Act']) #remove values for TOX under 0, impossible

print('Histogram of Toxicitiy (TOX)')
# add a 'best fit' line

plt.hist(df['Act'], bins=25)
plt.title(r'Histogram of TOX')
plt.ylabel('Frequency')
plt.xlabel('Lethal Dose 50%')
plt.savefig('figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig3_hist_Acts_vs_tox.png')


# # Standardize Data
scaler = preprocessing.StandardScaler()
df_scaled = df.drop(['MOLECULE', 'Act'], axis=1)
x_red = pd.DataFrame(scaler.fit_transform(df_scaled), columns=df_scaled.columns, index=df_scaled.index) 

X_train_all, X_test_all, y_train_all, y_test_all = train_test_split \
(x_red, df['Act'], test_size=0.2)#, random_state=2212) #.drop(['Act'])

X_train_all.columns

# # Remove Extraneous and Character Columns

#Y_train in data set suddenly contained missings when using pIC50, so using original version from EDA code
target = df['Act']
features =x_red.copy() #.drop(['TOX'], axis=1)
target.shape, features.shape

pd.DataFrame(x_red).to_csv(data_dir + 'X_all_act'+str(actfile)+'.csv')
pd.DataFrame(target).to_csv(data_dir + 'targets_act'+str(actfile)+'.csv')

# Create PC reduced version with all data
pca = PCA(n_components=nfeats)
pca_result = pca.fit_transform(features)

X_train_pca, X_test_pca, y_train_pca, y_test_pca = train_test_split \
(features, target, test_size=0.2)#, random_state=2212)


# # View correlations in Data, FULL DATA SET

# calculate the correlation matrix on the numeric columns
corr = x_red.select_dtypes('number').corr()

# plot the heatmap
sns.heatmap(corr)

#corr.iloc[:,-1]
features.shape[1]

kernel = 1.0 * RBF(length_scale=1e1, length_scale_bounds=(1e-2, 1e3)) + WhiteKernel(noise_level=1, noise_level_bounds=(1e-5, 1e1))


# # Create initial models with all data - Takes a long time due to number of columns

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
    plt.savefig('figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig6_nnet_Acts_vs_tox.png')

    #plt.show()
    
     
#    model.eval()
#    with torch.no_grad():
#        # Test out inference with 5 samples
#        for i in range(5):
#            X_sample = X_test_raw[i: i+1]
#            X_sample = scaler.transform(X_sample)
#            X_sample = torch.tensor(X_sample, dtype=torch.float32)
#            y_pred = model(X_sample)
#            print(f"{X_test_raw[i]} -> {y_pred[0].numpy()} (expected {y_test[i].numpy()})")

num_pipeline = Pipeline(steps=[('impute', SimpleImputer(strategy='mean'))])


def prepare_model(algorithm, X_train, y_train): 
    model = Pipeline(steps=[('preprocessing', num_pipeline),('algorithm', algorithm)])
    model.fit(X_train.loc[:,columns_to_keep], y_train)
    return model

algorithms = [
                ['RandomForestRegressor',RandomForestRegressor \
                 (random_state=42, n_estimators = 300, max_depth = 30, criterion = 'absolute_error',n_jobs=-1)], 
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

def runalgos(X_train, X_test, y_train, y_test):
    for algorithm in algorithms:
        name = algorithm[0]
        names.append(name)
        print(name)
        start_time = time.time()
        #model = algorithm[1]
        model = prepare_model(algorithm[1], X_train, y_train)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        end_time = time.time()
        times.append(end_time - start_time)
        mses.append(mean_squared_error(y_test, pred))
        maes.append(mean_absolute_error(y_test, pred))
        r2.append(r2_score(y_test, pred))
        print(name + ': ',mean_squared_error(y_test, pred))
    
    results_dict = {'Algorithm': names, 'MSE': mses, 'MAE': maes, 'R2': r2, 'Time': times}
    #results_sort = pd.DataFrame(results_dict).sort_values(by='MSE', ascending=1)
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

print("ALL Records Original Features Model Pipeline actfile="+str(actfile)+" version="+str(version))
columns_to_keep = X_train_all.columns

results_dict_all = runalgos(X_train_all, X_test_all, y_train_all, y_test_all)

pd.DataFrame.from_dict(results_dict_all, orient='index').T.sort_values(by='MSE', ascending=1).head(10)
bestmod(RandomForestRegressor, X_train_all, X_test_all, y_train_all, y_test_all)

# # SHAP with best model values
def shapgrad(model, X_train, y_train):
    rdr = model() #GradientBoostingRegressor()
    rdr.fit(X_train, y_train)
    explainer = shap.TreeExplainer(rdr, data=X_train)
    shap_values = explainer.shap_values(X_train, check_additivity=False)#, check_additivity=False)
    
    # visualize the model's dependence on the first feature
    shap.summary_plot(shap_values, X_train)
    feature_names = X_train.columns
    
    rf_resultX = pd.DataFrame(shap_values, columns = feature_names)#.values, columns = feature_names)
    vals = np.abs(rf_resultX.values).mean(0)
    
    shap_importance = pd.DataFrame(list(zip(feature_names, vals)),
                                      columns=['col_name','feature_importance_vals'])
    shap_importance.sort_values(by=['feature_importance_vals'],
                                   ascending=False, inplace=True)
    columns_to_keep = shap_importance.iloc[0:nfeats, 0].values.tolist()
    kfoldin = X_train.loc[:, columns_to_keep]
    #kfoldin['Act'] = y_train_all
    kfoldin['Act'] = y_train
    X_train_shap, X_test_shap, y_train_shap, y_test_shap = train_test_split \
    (kfoldin.loc[:, columns_to_keep], y_train, test_size=0.2)#, random_state=2212)

    shap.summary_plot(shap_values, X_train, plot_type='bar')
    plt.savefig('figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig5_shap_grad.png')

    return X_train_shap, X_test_shap, y_train_shap, y_test_shap, columns_to_keep, kfoldin

def shapkern(model, X_train, y_train):
    print("Starting SHAP analysis...")
    print_memory_usage()
    
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
    plt.savefig('figures/actfile_'+str(actfile)+'recs'+str(recs)+'fig5_shap_grad.png')
    
    # Clean up memory
    del shap_values, explainer
    gc.collect()
    
    return X_train_shap, X_test_shap, y_train_shap, y_test_shap, columns_to_keep, kfoldin

X_train_all.shape


# ## SHAPGRAD 
# Runs quickly but not always as well and only for Tree based models, not quite as powerful 
# ## SHAPKERN 
# Can be used for all model types and generally provides better results but is very slow.

x_trn_lim = X_train_all.iloc[:shapsamp,:]
print(X_train_all.shape)

print(x_trn_lim.shape)


y_trn_lim = y_train_all[:shapsamp]
print(len(y_train_all))
print(len(y_trn_lim))
print("ALL Records Original Features Model Pipeline"+str(recs)+" subsample #"+str(subnum)+' actfile='+str(actfile)+" version="+str(version))
print_memory_usage()  # Monitor memory before SHAP

# HEAVY 
X_train_shap, X_test_shap, y_train_shap, y_test_shap, columns_to_keep, kfoldin = shapkern(RandomForestRegressor,x_trn_lim, y_trn_lim) 
#X_train_shap, X_test_shap, y_train_shap, y_test_shap, columns_to_keep, kfoldin = shapkern(RandomForestRegressor,x_trn_lim, y_trn_lim) 
kfoldin.info()

pd.DataFrame(X_train_shap).to_csv(data_dir + 'merck_X_train_shap_act'+str(actfile)+'.csv')
pd.DataFrame(X_test_shap).to_csv(data_dir + 'merck_X_test_shap_act'+str(actfile)+'.csv')
pd.DataFrame(y_train_shap).to_csv(data_dir + 'merck_y_train_shap_act'+str(actfile)+'.csv')
pd.DataFrame(y_test_shap).to_csv(data_dir + 'merck_y_test_shap_act'+str(actfile)+'.csv')
pd.DataFrame(columns_to_keep).to_csv(data_dir + 'merck_shap_columns_to_keep_act'+str(actfile)+'.csv')
pd.DataFrame(kfoldin ).to_csv(data_dir + 'merck_kfoldin_shap_act'+str(actfile)+'.csv')


# Create backup so we don't need to run SHAP again to get kfoldin
kfold_back = kfoldin.copy()
columns_to_keep_back = columns_to_keep

kfoldin.columns

kfoldin.shape

#samp_size = int(subsamp_size/n_clusters)


km = KMeans(n_clusters=n_clusters)
visualizer = KElbowVisualizer(km, k=(1,10))

visualizer.fit(kfold_back)   # Fit the data to the visualizer
visualizer.show()            # Finalize and render the figure


shap_inputs = x_red.copy()
shap_inputs['Act'] = df['Act']

shap_input_samp = shap_inputs.sample(n=shapsamp) 

x_shap_input_samp = shap_input_samp.drop(['Act'], axis=1)
y_shap_input_samp = shap_input_samp['Act']

def subsamp(data, n, iteration, replc=True, columns_to_keep=columns_to_keep):
    #targs = data['TOX']
      
    clusdat=data.drop('Act', axis=1) #REMOVE TARGETS SO WE ARE NOT CLUSTERING ON THEM   #[df_sub.index)
    print(clusdat.info)
    km = KMeans(n_clusters=n_clusters) #, random_state=1)
    new = clusdat._get_numeric_data().dropna(axis=1)
    print(new.info)
    km.fit(new)
    predict=km.predict(new)
    
    samp_size = int(n/n_clusters)
    print(samp_size)
    #df_test,[kinfo['cluspred']== i].sample(1)
    data['cluspred'] = pd.Series(predict, index=clusdat.index)
    #clusdat['TOX'] = data['TOX']
    df_sub = data.groupby('cluspred').sample(n=samp_size, replace=replc) #.sample(n=samp_size, replace=False)

    # Ensure unique indices for both data and df_sub
    data.reset_index(drop=True, inplace=True)
    df_sub.reset_index(drop=True, inplace=True)

    # Now perform the isin operation
    data = data[~data.isin(df_sub)].dropna(how='all')
    #clus=clusdat.drop(df_sub.index)
    sub_targ = df_sub['Act']
    clus_pred = df_sub['cluspred']
    sub_feat = df_sub.loc[:, columns_to_keep]#.drop(['TOX'], axis=1)  
    sub_feat['cluspred'] = clus_pred
    #sub_feat['cluspred'] = pd.Series(predict, index=sub_feat.index)
    
    X_train_sub, X_test_sub, y_train_sub, y_test_sub = train_test_split \
    (sub_feat, sub_targ, test_size=0.2, stratify=sub_feat['cluspred'])#, random_state=2212)
    
    print('>Original data size',len(data), '>DF_sub'+str(iteration)+' Length', len(df_sub), '>Number removed from original data', n)
    
    pd.DataFrame(X_train_sub).to_csv(data_dir + 'X_train_'+str(n)+'rec_sub'+str(iteration)+'act'+str(actfile)+'v'+str(version)+'.csv')
    pd.DataFrame(X_test_sub).to_csv(data_dir + 'X_test_'+str(n)+'rec_sub'+str(iteration)+'act'+str(actfile)+'v'+str(version)+'.csv')
    pd.DataFrame(y_train_sub).to_csv(data_dir + 'y_train_'+str(n)+'rec_sub'+str(iteration)+'act'+str(actfile)+'v'+str(version)+'.csv')
    pd.DataFrame(y_test_sub).to_csv(data_dir + 'y_test_'+str(n)+'rec_sub'+str(iteration)+'act'+str(actfile)+'v'+str(version)+'.csv')
    print('Output of Act File: ', actfile)

    names = []
    times = []
    mses = []
    maes = []
    r2 = []
    columns_to_keep = X_train_sub.columns
    results_dict_shap = runalgos(X_train_sub, X_test_sub, y_train_sub, y_test_sub)
    #results_dict_shap = runalgos(X_train_sub,X_test_sub, )
    #display(pd.DataFrame(results_dict_shap).sort_values(by='MSE', ascending=1))

    pd.set_option('display.max_rows', None)
    
    print("Subsample SHAP Model Pipeline for recs="+str(recs)+" subsample #"+str(subnum)+' actfile='+str(actfile)+" version="+str(version))
    pd.DataFrame.from_dict(results_dict_shap, orient='index').T.sort_values(by='MSE', ascending=1).head(10)
    bestmod(RandomForestRegressor, X_train_sub, X_test_sub, y_train_sub, y_test_sub)
    #nnet(df_train, df_test, df_ytrn, df_ytst)
    return X_train_sub, X_test_sub, y_train_sub, y_test_sub, kfoldin

version = 5
shap_back = shap_inputs.copy()
ecs = 100
print(shap_inputs.info)

#%%
X_train_sub1, X_test_sub1, y_train_sub1, y_test_sub1, kfoldin = subsamp(shap_inputs, recs, 1, columns_to_keep=columns_to_keep)
X_train_sub2, X_test_sub2, y_train_sub2, y_test_sub2, kfoldin = subsamp(shap_inputs, recs, 2, columns_to_keep=columns_to_keep)
X_train_sub3, X_test_sub3, y_train_sub3, y_test_sub3, kfoldin = subsamp(shap_inputs, recs, 3, columns_to_keep=columns_to_keep)
X_train_sub4, X_test_sub4, y_train_sub4, y_test_sub4, kfoldin = subsamp(shap_inputs, recs, 4, columns_to_keep=columns_to_keep)
X_train_sub5, X_test_sub5, y_train_sub5, y_test_sub5, kfoldin = subsamp(shap_inputs, recs, 5, columns_to_keep=columns_to_keep)

X_train_sub6, X_test_sub6, y_train_sub6, y_test_sub6, kfoldin = subsamp(shap_inputs, recs, 6, columns_to_keep=columns_to_keep)
X_train_sub7, X_test_sub7, y_train_sub7, y_test_sub7, kfoldin = subsamp(shap_inputs, recs, 7, columns_to_keep=columns_to_keep)
X_train_sub8, X_test_sub8, y_train_sub8, y_test_sub8, kfoldin = subsamp(shap_inputs, recs, 8, columns_to_keep=columns_to_keep)
X_train_sub9, X_test_sub9, y_train_sub9, y_test_sub9, kfoldin = subsamp(shap_inputs, recs, 9, columns_to_keep=columns_to_keep)
X_train_sub10, X_test_sub10, y_train_sub10, y_test_sub10, kfoldin = subsamp(shap_inputs, recs, 10, columns_to_keep=columns_to_keep)

X_train_sub11, X_test_sub11, y_train_sub11, y_test_sub11, kfoldin = subsamp(shap_inputs, recs, 11, columns_to_keep=columns_to_keep)
X_train_sub12, X_test_sub12, y_train_sub12, y_test_sub12, kfoldin = subsamp(shap_inputs, recs, 12, columns_to_keep=columns_to_keep)
X_train_sub13, X_test_sub13, y_train_sub13, y_test_sub13, kfoldin = subsamp(shap_inputs, recs, 13, columns_to_keep=columns_to_keep)
X_train_sub14, X_test_sub14, y_train_sub14, y_test_sub14, kfoldin = subsamp(shap_inputs, recs, 14, columns_to_keep=columns_to_keep)
X_train_sub15, X_test_sub15, y_train_sub15, y_test_sub15, kfoldin = subsamp(shap_inputs, recs, 15, columns_to_keep=columns_to_keep)

X_train_sub16, X_test_sub16, y_train_sub16, y_test_sub16, kfoldin = subsamp(shap_inputs, recs, 16, columns_to_keep=columns_to_keep)
X_train_sub17, X_test_sub17, y_train_sub17, y_test_sub17, kfoldin = subsamp(shap_inputs, recs, 17, columns_to_keep=columns_to_keep)
X_train_sub18, X_test_sub18, y_train_sub18, y_test_sub18, kfoldin = subsamp(shap_inputs, recs, 18, columns_to_keep=columns_to_keep)
X_train_sub19, X_test_sub19, y_train_sub19, y_test_sub19, kfoldin = subsamp(shap_inputs, recs, 19, columns_to_keep=columns_to_keep)
X_train_sub20, X_test_sub20, y_train_sub20, y_test_sub20, kfoldin = subsamp(shap_inputs, recs, 20, columns_to_keep=columns_to_keep)

X_train_sub21, X_test_sub21, y_train_sub21, y_test_sub21, kfoldin = subsamp(shap_inputs, recs, 21, columns_to_keep=columns_to_keep)
X_train_sub22, X_test_sub22, y_train_sub22, y_test_sub22, kfoldin = subsamp(shap_inputs, recs, 22, columns_to_keep=columns_to_keep)
X_train_sub23, X_test_sub23, y_train_sub23, y_test_sub23, kfoldin = subsamp(shap_inputs, recs, 23, columns_to_keep=columns_to_keep)
X_train_sub24, X_test_sub24, y_train_sub24, y_test_sub24, kfoldin = subsamp(shap_inputs, recs, 24, columns_to_keep=columns_to_keep)
#%%
X_train_sub25, X_test_sub25, y_train_sub25, y_test_sub25, kfoldin = subsamp(shap_inputs, recs, 25, columns_to_keep=columns_to_keep)

# shap_inputs = shap_back.copy()
# recs = 200
# X_train_sub1, X_test_sub1, y_train_sub1, y_test_sub1, kfoldin = subsamp(shap_inputs, recs, 1, columns_to_keep=columns_to_keep)
# X_train_sub2, X_test_sub2, y_train_sub2, y_test_sub2, kfoldin = subsamp(shap_inputs, recs, 2, columns_to_keep=columns_to_keep)
# X_train_sub3, X_test_sub3, y_train_sub3, y_test_sub3, kfoldin = subsamp(shap_inputs, recs, 3, columns_to_keep=columns_to_keep)
# X_train_sub4, X_test_sub4, y_train_sub4, y_test_sub4, kfoldin = subsamp(shap_inputs, recs, 4, columns_to_keep=columns_to_keep)
# X_train_sub5, X_test_sub5, y_train_sub5, y_test_sub5, kfoldin = subsamp(shap_inputs, recs, 5, columns_to_keep=columns_to_keep)

# shap_inputs = shap_back.copy()
# recs = 800
# X_train_sub1, X_test_sub1, y_train_sub1, y_test_sub1, kfoldin = subsamp(shap_inputs, recs, 1, columns_to_keep=columns_to_keep) #, replc=False, columns_to_keep=columns_to_keep)
# X_train_sub2, X_test_sub2, y_train_sub2, y_test_sub2, kfoldin = subsamp(shap_inputs, recs, 2, columns_to_keep=columns_to_keep) #, replc=False, columns_to_keep=columns_to_keep)
# X_train_sub3, X_test_sub3, y_train_sub3, y_test_sub3, kfoldin = subsamp(shap_inputs, recs, 3, columns_to_keep=columns_to_keep) #, replc=False, columns_to_keep=columns_to_keep)
# X_train_sub4, X_test_sub4, y_train_sub4, y_test_sub4, kfoldin = subsamp(shap_inputs, recs, 4, columns_to_keep=columns_to_keep) #, replc=False, columns_to_keep=columns_to_keep)
# X_train_sub5, X_test_sub5, y_train_sub5, y_test_sub5, kfoldin = subsamp(shap_inputs, recs, 5, columns_to_keep=columns_to_keep) #, replc=False, columns_to_keep=columns_to_keep)
# %%
