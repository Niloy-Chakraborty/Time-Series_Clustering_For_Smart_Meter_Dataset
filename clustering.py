"""
Script : clustering.py
Definition: This script is written for the following functionality :
    1. Dimensionality reduction
    2. K-means clustering.
    3. Hierarchical clustering.
    4. Creation of cluster wise dictionary.
    5. Cluster wise acorn data mapping
"""

# import libraries
import pandas as pd  # version 0.23.4 , because of some bugs in the latest version 0.24
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from sklearn.cluster import AgglomerativeClustering
from math import sqrt
from sklearn.manifold import TSNE
import seaborn as sns
from sklearn.decomposition import PCA
from collections import OrderedDict
import data_process
import datetime as dt
from scipy.stats import pearsonr
from sklearn.metrics.pairwise import pairwise_distances
from scipy.cluster.hierarchy import fcluster

import scipy.cluster.hierarchy as hac




"""
Function :  dimension_reduction
Definition: this function takes 'df1' as argument, and does the following:
            1. reduce the feature dimension to 2 or 3 ; using TSNE or PCA.

Return: dataFrame "data"

"""


def dimension_reduction(df):
    data = TSNE(n_components=3, perplexity=40).fit_transform(df)
    #data = PCA(n_components=3).fit_transform(df)
    print(data.shape)
    return data


"""
Function : clustering
Definition: this function takes the processed dataset 'df1' as argument, and does the following:
            1. creates Kmeans model
            2. fit the model with the dataset 'df1
            3. save the labels named as 'labels.csv'
            4. Save the dataset after the above tasks , named 'influxData_Clustered.csv' 
Return: df1, labels

"""


def clustering_kmeans(df1, n_cluster):
    n_clusters = n_cluster
    kmeans = KMeans(n_clusters, init='k-means++', n_init=10, max_iter=500, algorithm='auto', verbose=0)

    kmeans.fit(df1)
    labels = kmeans.labels_

    # df1["labels"] = labels
    # pd.DataFrame(labels).to_csv("labels.csv", header="label",index=0)

    # print(df1.info())
    # df1.to_csv("influxData_Clustered.csv")

    return labels


"""
Function : DTWDistance
Definition: this function takes s1,s2 as argument, and does the following:
            1. performs Dynamic time wrapping distance between data points 

Return: the distance measures. 

"""


def DTWDistance(s1, s2):
    # print("DTW called")
    DTW = {}

    for i in range(len(s1)):
        DTW[(i, -1)] = float('inf')
    for i in range(len(s2)):
        DTW[(-1, i)] = float('inf')
    DTW[(-1, -1)] = 0

    for i in range(len(s1)):
        for j in range(len(s2)):
            dist = (s1[i] - s2[j]) ** 2
            DTW[(i, j)] = dist + min(DTW[(i - 1, j)], DTW[(i, j - 1)], DTW[(i - 1, j - 1)])

    return sqrt(DTW[len(s1) - 1, len(s2) - 1])


def pearson_affinity(a,b):
   metric = 1 - np.array([pearsonr(a,b)][0])
   #print ("metric is",metric)

   return metric[0]


"""
Function : clustering_hierarchial
Definition: this function takes processed df as argument, and does the following:
            1. performs hierarchical clustering using custom DTW distance 
            2. plot the denograms and shows the correlation.

Return: the distance measures. 

"""



def clustering_hierarchial(df, n_cluster):

    m = pairwise_distances(df, df, metric=pearson_affinity)

    linked = linkage(df, method='complete', metric=pearson_affinity)  # , metric= DTWDistance/pearson_affinity

    clustering = AgglomerativeClustering(n_clusters=n_cluster, affinity='precomputed', linkage='average')
    clustering = clustering.fit(m)
    labels = clustering.labels_

    print(labels)
    print("inside hierarchial clustering")

    plt.figure(figsize=(15, 11))
    dendrogram(linked,
               orientation='top',
               distance_sort='descending',
               show_leaf_counts=True)
    plt.show()
    return labels


"""
Function : cluster_dictionary
Definition: this function takes processed df and labels as arguments, and creates dictionary of clusters where the 
            cluster labels are represented by the keys.

Return: dictionary of the Meter IDs where the key is the cluster number.

"""


def cluster_dictionary(df1, labels):
    df1 = df1.set_index("time")
    labels = labels.tolist()
    dict = {}

    for j in range(len(set(labels))):  # use set() to remove duplicates of list

        lis = []
        for i in range(len(labels)):
            # print("i", labels[i])
            if labels[i] == j:

                lis.append(df1.columns[i])
            else:
                pass
        dict[str(j)] = lis  # dictionary of clusters, key= label no and value = list of clusters
    print(dict)
    return dict


"""
Function : acorn_map
Definition: Function for mapping the acorn groups with the meter IDs and the cluster labels
Return: labeled acorn dataset

"""


def acorn_map(main_df, labels, data_choice):
    df = pd.DataFrame()
    if data_choice==2:
        df["HouseholdID"] = main_df.Household_Name.str[17:] #for subset data
        acorn_data = pd.read_csv("LondonSmartMeter_Info.txt", sep='\t')  # for subset
    else:
        df["HouseholdID"] = main_df["Household_Name"] # for full data
        acorn_data = pd.read_csv("acorn_df.csv")  # for full dataset
        acorn_data = acorn_data.reset_index()
        acorn_data = acorn_data.rename(index=str, columns={"Unnamed: 0": "HouseholdID"})

    df["label"] = labels

    df = df.astype(str)
    print(df.info())



    #acorn_data= acorn_data.rename({"":"HouseholdID"},axis=1)

    acorn_data = acorn_data.astype(str)
    print("hi acorn data", acorn_data.head())
    df = pd.merge(df, acorn_data, on=["HouseholdID"], how='inner')
    df = df.groupby(["label"]).apply(lambda x: x.reset_index(drop=True))
    print(df.head)
    print(df.info)

    df = df.drop(df.columns[1], axis=1)

    print(df.info())

    #df.to_csv("acorn_map.csv")
    return df


def correlation(df1, dict_cluster):

    radiation_df = pd.read_csv("data_london_global_radiation2013.txt", sep=';')
    print(radiation_df.head())
    #radiation_df.plot()
    radiation_df["Date at end of observation period"] = pd.to_datetime(radiation_df["Date at end of observation period"], format='%Y-%m-%d')

    radiation_df = radiation_df.set_index("Date at end of observation period")
    radiation_df = radiation_df["Global solar irradiation amount [KJ/m2]"].resample("H").sum()
    #radiation_df.plot()

    df1['time'] = pd.to_datetime(df1['time'])
    df1 = df1.set_index("time")

    #df1 = df1[df1['time'].dt.year == 2013]
    print(df1.info())

    #plt.show()
    dict_correlation= {}
    j = 0
    for lis_cluster in dict_cluster.keys():

        lis=[]
        for i in dict_cluster[lis_cluster]:
            df1[i] = df1[i].resample("H").sum()

            #df1[i].plot()
            #plt.show()
            #exit()

            corr = df1[i].corr(radiation_df)
            lis.append(corr)
        dict_correlation["cluster"+str(lis_cluster)] = lis
        j=j+1

    print(dict_correlation)
    return dict_correlation



























