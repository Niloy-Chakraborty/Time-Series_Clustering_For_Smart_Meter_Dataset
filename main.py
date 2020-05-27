import data_process
import clustering
import Visualization
import pandas as pd
import clustering_autoencoder
from pathlib import Path
import os
from sklearn.model_selection import train_test_split
from numpy.random import seed


"""
Function : main

Definition: Function for automatic invocation and calling other functions.
Return: 

"""

if __name__ == '__main__':

    # Uncomment the next commented part for the initial separate data creation.
    # This code section creates two separate dataset for the Meter Consumption and the Acorn Distribution of the meters.
    '''
    try:
        data = pd.read_csv("LondonSmartMeter_Data_filtered2013nonull.tsv", sep="\t")
    except:
        pass

    print(data.info())
    acorn_df = data.iloc[:3].T
    acorn_df.columns = acorn_df.iloc[0]
    acorn_df = acorn_df.iloc[2:]

    print(acorn_df.head(10))
    print(acorn_df.head(10))

    acorn_df.to_csv("acorn_df.csv")
    df = data.iloc[4:]
    df = df.rename(index=str, columns={"HouseholdID": "time"})

    print(df.head())

    df.to_csv("London_full_data.csv")

    exit()
    '''


    df1 = pd.read_csv("London_full_data.csv")  # for London dataset
    df1 = df1.drop("Unnamed: 0", axis=1)
    df1.to_csv("df1.csv")

    print("YOUR DATASET LOOKS LIKE THIS:")
    print(df1.head(10))

    df3, df4, df5 = data_process.create_new_features(df1)
    Main_df = data_process.create_main_dataFrame(df1, df3, df4, df5)
    print("AFTER SOME PRE-PROCESSING, YOUR DATASET LOOKS LIKE THE FOLLOWING:")
    print(Main_df.head(10))

    data_choice="london_data"
    processed_df = data_process.data_preprocessing(Main_df, data_choice)


    #Comment the following line and uncomment the above part, if there is need for data
    #processed_df = pd.read_csv("dataset_preprocessed_" + str(data_choice) + ".csv")
    #df_reduced_dimension = processed_df

    df_reduced_dimension = clustering.dimension_reduction(processed_df)

    # Take first 1000 entries for training of k-Means and Hierarchical Clusteringg, leave the rest for validation
    df_reduced_dimension = df_reduced_dimension[:1000]

    print("DIMENSION OF THE DATASET IS REDUCED FOR FAST PROCESSING! NOW THE DATASET LOOKS LIKE THE FOLLOWING")
    print(pd.DataFrame(df_reduced_dimension).head(5))

    print("DATA PRE-PROCESSING COMPLETE, LET'S START CLUSTERING NOW :) ")

    print("#########################################################################################################")

    n_clusters = int(input("Enter the number of clusters: "))

    choice = int(input("1) Enter 1 for applying Kmeans on the dataset \n or"
                       "\n2)Enter 2 for applying Hierarchical clustering on the dataset: \n"
                       "\n3)Enter 3 for applying Autoencoder + KMeans clustering on the dataset: \n"
                       "\n4)Enter 4 for applying DBSCAN clustering on the dataset: \n"))

    print("#########################################################################################################")

    if choice == 1:
        labels = clustering.clustering_kmeans(df_reduced_dimension, n_clusters)  #df_reduced_dimension
        choice= "kmeans"
        print(labels)
    elif choice == 2:
        labels = clustering.clustering_hierarchial(df_reduced_dimension, n_clusters)
        choice = "hierarchical"
        print(labels)
    elif choice == 3:
        encoded_train, encoded_test = clustering_autoencoder.autoencoder()

        labels = clustering.clustering_kmeans(encoded_train, n_clusters)  # df_reduced_dimension
        choice = "autoencoder"

        print("length of labels: ",len(labels))

        # pd.DataFrame(labels).to_csv("labels.csv")
    elif choice == 4:
        labels = clustering.clustering_DBSCAN(df_reduced_dimension, n_clusters)
        choice = "DBSCAN"
        print(labels)
        # pd.DataFrame(labels).to_csv("labels.csv")
    else:
        print("wrong input!!!")

    pd.DataFrame(labels).to_csv("labels.csv")
    print("length of labels: ",len(labels))

    ########### VISUALIZATION OF THE DATASET AND CLUSTERED DATA##########

    Main_df, Main_df2= Main_df[:1000], Main_df[1001:]
    dict_cluster = clustering.cluster_dictionary(df1, labels)

    # Uncomment Next line for Acorn Group wise original Data Visualization
    #isualization.acornWisePlot(df1)
    # Uncomment Next line for Acorn Category wise original Data Visualization
    #Visualization.acornCategoryWisePlot(df1)

    # Uncomment Next line for general Data Visualization
    # Visualization.visualise_data(dict_cluster, df1,n_clusters)

    # Uncomment Next line for Clustered Data Visualization through Subplots
    # Visualization.cluster_visualisation_subplot(df1, labels, choice, n_clusters)

    # Uncomment Next line for Clustered Data Visualization through Single Plot
    # Visualization.cluster_visualisation_single_plot(dict_cluster, df1, choice, n_clusters)

    # Uncomment Next line for Mean Clustered Data Visualization1
    # df_mean = Visualization.mean_cluster_visualisation(dict_cluster, df1, choice, n_clusters)

    # Uncomment Next line for Resampled Clustered Data Visualization
    # Visualization.resampled_cluster_visualisation(choice,n_clusters)

    # Uncomment Next lines for Visualization of Acorn Distribution with the Clustered Data
    acorn_df = clustering.acorn_map(Main_df, labels, data_choice)
    Visualization.acorn_plot(acorn_df, n_clusters,choice)

    # Uncomment Next lines for average daily and hourly plot (Summer and Winter data in two different plot)
    # of the clustered data
    Visualization.Hourlyplot(choice,n_clusters)
    # Visualization.DayWiseHourlyplot(choice, n_clusters)

    # Uncomment Next lines for Visualization of box plot correlation of the radiation pattern with the Clustered Data
    dict_correlation = clustering.correlation(df1, dict_cluster)
    Visualization.correlation_plot(dict_correlation, n_clusters, choice)

    # Uncomment Next line for Visualization of weekly distribution of all the clusters in a single plot
    Visualization.dayWiseGrouppedPlot(choice,n_clusters)